# AOT ID: ['10_forward']
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


# kernel path: inductor_cache/xk/cxkaxztltouhwtswddb6rnpxa5oonqovm7g3iqjgkrqbwvicrh3q.py
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
    size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
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


# kernel path: inductor_cache/ps/cpsdtp7yo3kdfpx52nke5kwegrhl23hgbihwzje6lkjggs46el6j.py
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/a5/ca5ppr6qoodify4ggggoby3wotliss2ye42qh7lklla4sa2xhzzq.py
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
    ynumel = 2304
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 24)
    y1 = yindex // 24
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 24*x2 + 216*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/vh/cvh4vpv3awgcoxinmu5ervy6wedhiowlzf5obklomowpvfgsuv3g.py
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


# kernel path: inductor_cache/pu/cpuc4vsymd5oggptzhrlh5akns3we2fr2i3vrolormczttro6f2s.py
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
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 80)
    y1 = yindex // 80
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 80*x2 + 720*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vi/cvihmyet27v7u7rah7imvgfdasaxoyyrxqqhldlykefcvak6gq4b.py
# Topologically Sorted Source Nodes: [features_0_1, sigmoid_1, mul_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_0_1 => add_1, mul_1, mul_2, sub
#   mul_1 => mul_3
#   sigmoid_1 => sigmoid
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_1,), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/qr/cqrx3g4hgdqoopma2evirruvfifkyk6t7xvm2bpsre7vjmy6cuxs.py
# Topologically Sorted Source Nodes: [features_1_conv_4, add_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_6
#   features_1_conv_4 => add_5, mul_10, mul_9, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %add_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/i7/ci7m7mtrjtyzs5adlxcrgrpxf5cdbtjoet2jf27zzkahjkdxgmmq.py
# Topologically Sorted Source Nodes: [features_4_conv_1, sigmoid_5, mul_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_4_conv_1 => add_18, mul_26, mul_27, sub_7
#   mul_5 => mul_28
#   sigmoid_5 => sigmoid_4
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_61), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_63), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_18,), kwargs = {})
#   %mul_28 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %sigmoid_4), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 96)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/d7/cd75ts46koup27g4tlh3yt7dh7k7zo6okvyv3okp2e2ih6dolu6q.py
# Topologically Sorted Source Nodes: [features_4_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_4_conv_4 => add_20, mul_30, mul_31, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %unsqueeze_69), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_71), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 48)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/kf/ckfpeltu3ortcc26nd3sfvvd475fni2h635gk7cu5wn5g2kwxzr5.py
# Topologically Sorted Source Nodes: [features_5_conv_1, sigmoid_6, mul_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_5_conv_1 => add_22, mul_33, mul_34, sub_9
#   mul_6 => mul_35
#   sigmoid_6 => sigmoid_5
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_33, %unsqueeze_77), kwargs = {})
#   %add_22 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %unsqueeze_79), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_22,), kwargs = {})
#   %mul_35 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_22, %sigmoid_5), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xk/cxkgjvtoitf6rezcpeq6b7sdv4lsr6kk33qqu26osxiiub5hltez.py
# Topologically Sorted Source Nodes: [features_5_conv_4, add_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_25
#   features_5_conv_4 => add_24, mul_37, mul_38, sub_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_85), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_87), kwargs = {})
#   %add_25 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %add_24), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 48)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/5f/c5f6ksgt6njp4k3ywrcnjkytsjafvzbitf6byfknwgavq7zjglrw.py
# Topologically Sorted Source Nodes: [features_9_conv_1, sigmoid_10, mul_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_9_conv_1 => add_42, mul_61, mul_62, sub_17
#   mul_10 => mul_63
#   sigmoid_10 => sigmoid_9
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_141), kwargs = {})
#   %add_42 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_143), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_42,), kwargs = {})
#   %mul_63 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_42, %sigmoid_9), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/dj/cdjt7vux4tnsnofgmwazlxbhzd6u2mgukqxoetepsq7lmj6epsgc.py
# Topologically Sorted Source Nodes: [features_9_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_9_conv_4 => add_44, mul_65, mul_66, sub_18
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_65, %unsqueeze_149), kwargs = {})
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_66, %unsqueeze_151), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 80)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/w2/cw2hkuklrkknwpr64b52xad5vwvgbmvzkmosjhufvtw74bgrljml.py
# Topologically Sorted Source Nodes: [features_10_conv_1, sigmoid_11, mul_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_10_conv_1 => add_46, mul_68, mul_69, sub_19
#   mul_11 => mul_70
#   sigmoid_11 => sigmoid_10
# Graph fragment:
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_19, %unsqueeze_153), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_155), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_68, %unsqueeze_157), kwargs = {})
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_69, %unsqueeze_159), kwargs = {})
#   %sigmoid_10 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_46,), kwargs = {})
#   %mul_70 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_46, %sigmoid_10), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/he/cheypx4fxq5u7iqyq2he5yktnztnwgrcg3up46myaajede4bhy67.py
# Topologically Sorted Source Nodes: [features_10_conv_4, add_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_8 => add_49
#   features_10_conv_4 => add_48, mul_72, mul_73, sub_20
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_72, %unsqueeze_165), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_73, %unsqueeze_167), kwargs = {})
#   %add_49 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_44, %add_48), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 80)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ca/ccaqq3unqn53pmsdkygairopsv4tx3waokzxgrmql4v26t2q2fu3.py
# Topologically Sorted Source Nodes: [features_14_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_14_conv_4 => add_68, mul_100, mul_101, sub_28
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_229), kwargs = {})
#   %add_68 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_231), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/np/cnp2m2kwmgldlef6qbuqskzggozlh26caro5vjkw7fte3y6s4px6.py
# Topologically Sorted Source Nodes: [sigmoid_16, mul_16, features_14_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_14_conv_6_avg_pool => mean
#   mul_16 => mul_102
#   sigmoid_16 => sigmoid_15
# Graph fragment:
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_68,), kwargs = {})
#   %mul_102 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_68, %sigmoid_15), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_102, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_17 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_17(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1280
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 320)
    x1 = xindex // 320
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 320*r2 + 5120*x1), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 16.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yv/cyvl3mdqh3hxpnpiyxsrfcqraxmmshvozd3i3qdmikslzmsktcta.py
# Topologically Sorted Source Nodes: [sigmoid_17, mul_17], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_17 => mul_103
#   sigmoid_17 => sigmoid_16
# Graph fragment:
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm,), kwargs = {})
#   %mul_103 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, %sigmoid_16), kwargs = {})
triton_poi_fused_mul_sigmoid_18 = async_compile.triton('triton_poi_fused_mul_sigmoid_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eu/ceuzzqtfk2kj3bncb7fuwjp6ym5nmjjukqhuhzgzj7cwsa66ncf4.py
# Topologically Sorted Source Nodes: [sigmoid_16, mul_16, mul_18], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_16 => mul_102
#   mul_18 => mul_104
#   sigmoid_16 => sigmoid_15
# Graph fragment:
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_68,), kwargs = {})
#   %mul_102 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_68, %sigmoid_15), kwargs = {})
#   %mul_104 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_102, %view_1), kwargs = {})
triton_poi_fused_mul_sigmoid_19 = async_compile.triton('triton_poi_fused_mul_sigmoid_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 320)
    x2 = xindex // 5120
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 320*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/xi/cxisijlf2w767cf7cpehz2jtp45t5s56ac2llchpq2dcvcqzrvgu.py
# Topologically Sorted Source Nodes: [features_14_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_14_conv_8 => add_70, mul_106, mul_107, sub_29
# Graph fragment:
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_106, %unsqueeze_237), kwargs = {})
#   %add_70 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_107, %unsqueeze_239), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2u/c2upva4cwqhfxysgijyt3svjkbxjfwnfx4dxugnkm3dhbo6gfevv.py
# Topologically Sorted Source Nodes: [features_15_conv_1, sigmoid_20, mul_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_15_conv_1 => add_72, mul_109, mul_110, sub_30
#   mul_19 => mul_111
#   sigmoid_20 => sigmoid_18
# Graph fragment:
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_245), kwargs = {})
#   %add_72 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_247), kwargs = {})
#   %sigmoid_18 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_72,), kwargs = {})
#   %mul_111 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_72, %sigmoid_18), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/vb/cvbkoomir2wnt2kgvj3rqyorz4zxdt2b3obnbo5qywn2vdp2drrw.py
# Topologically Sorted Source Nodes: [features_15_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_15_conv_4 => add_74, mul_113, mul_114, sub_31
# Graph fragment:
#   %sub_31 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_31, %unsqueeze_249), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_31, %unsqueeze_251), kwargs = {})
#   %mul_114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_113, %unsqueeze_253), kwargs = {})
#   %add_74 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_114, %unsqueeze_255), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/2n/c2nuw3ika64hmv3umanizx4yob4eyidlqsaqsel7ncy67csq5kay.py
# Topologically Sorted Source Nodes: [sigmoid_22, mul_20, features_15_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_15_conv_6_avg_pool => mean_1
#   mul_20 => mul_115
#   sigmoid_22 => sigmoid_19
# Graph fragment:
#   %sigmoid_19 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_115 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_19), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_115, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_23 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_23(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 640)
    x1 = xindex // 640
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 640*r2 + 10240*x1), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 16.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a5/ca55nyjtjvhaoissd45p6jwdaj5dlrikxhlkazbo3huozn37usi6.py
# Topologically Sorted Source Nodes: [sigmoid_24, mul_21], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_21 => mul_116
#   sigmoid_24 => sigmoid_20
# Graph fragment:
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_2,), kwargs = {})
#   %mul_116 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_2, %sigmoid_20), kwargs = {})
triton_poi_fused_mul_sigmoid_24 = async_compile.triton('triton_poi_fused_mul_sigmoid_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ow/cowx4jwliajfru6ta2ahg4q2o2gb6zjfs6fbx3serdkqiggek27q.py
# Topologically Sorted Source Nodes: [sigmoid_22, mul_20, mul_22], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_20 => mul_115
#   mul_22 => mul_117
#   sigmoid_22 => sigmoid_19
# Graph fragment:
#   %sigmoid_19 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_74,), kwargs = {})
#   %mul_115 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_74, %sigmoid_19), kwargs = {})
#   %mul_117 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %view_3), kwargs = {})
triton_poi_fused_mul_sigmoid_25 = async_compile.triton('triton_poi_fused_mul_sigmoid_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 40960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 640)
    x2 = xindex // 10240
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 640*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/rz/crzrqeizg25kk6ymb3vd34ve2ojyk54ji75rz6mibmqvu2scszj2.py
# Topologically Sorted Source Nodes: [features_15_conv_8, add_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_12 => add_77
#   features_15_conv_8 => add_76, mul_119, mul_120, sub_32
# Graph fragment:
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_32, %unsqueeze_257), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %unsqueeze_259), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %unsqueeze_261), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %unsqueeze_263), kwargs = {})
#   %add_77 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %add_76), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ff/cffgwd7u56ahdmu4bda25ewl77bi4vtrjqfn27hmeedjjuqf2fva.py
# Topologically Sorted Source Nodes: [features_21_conv_1, sigmoid_62, mul_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_21_conv_1 => add_114, mul_187, mul_188, sub_48
#   mul_43 => mul_189
#   sigmoid_62 => sigmoid_42
# Graph fragment:
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %unsqueeze_385), kwargs = {})
#   %mul_187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_387), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_187, %unsqueeze_389), kwargs = {})
#   %add_114 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_188, %unsqueeze_391), kwargs = {})
#   %sigmoid_42 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_114,), kwargs = {})
#   %mul_189 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_114, %sigmoid_42), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 960)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/4c/c4cue6aewmmcui276wjbvfzrtlujb4xbvuirnv5bdsanltjge3vy.py
# Topologically Sorted Source Nodes: [features_21_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_21_conv_4 => add_116, mul_191, mul_192, sub_49
# Graph fragment:
#   %sub_49 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_49, %unsqueeze_393), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_49, %unsqueeze_395), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_191, %unsqueeze_397), kwargs = {})
#   %add_116 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_192, %unsqueeze_399), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_28 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 960)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/7n/c7ncxer3cnr67x5oilvaoidrjqxtyd2rl7a5v2dz6kxjmqqkw4b6.py
# Topologically Sorted Source Nodes: [sigmoid_64, mul_44, features_21_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_21_conv_6_avg_pool => mean_7
#   mul_44 => mul_193
#   sigmoid_64 => sigmoid_43
# Graph fragment:
#   %sigmoid_43 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_116,), kwargs = {})
#   %mul_193 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %sigmoid_43), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_193, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_29 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4096, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_29(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 960)
    x1 = xindex // 960
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 960*r2 + 15360*x1), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 16.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k5/ck52fznshon5fnpyxhjlkncvu7bsp7v2wtgvxv2phndisn6l3ygk.py
# Topologically Sorted Source Nodes: [sigmoid_64, mul_44, mul_46], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_44 => mul_193
#   mul_46 => mul_195
#   sigmoid_64 => sigmoid_43
# Graph fragment:
#   %sigmoid_43 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_116,), kwargs = {})
#   %mul_193 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_116, %sigmoid_43), kwargs = {})
#   %mul_195 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %view_15), kwargs = {})
triton_poi_fused_mul_sigmoid_30 = async_compile.triton('triton_poi_fused_mul_sigmoid_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 960)
    x2 = xindex // 15360
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 960*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/df/cdfcganxz6pjmoz6yjenupdh27hdluoodhimlig5ul7n6dvt6wm3.py
# Topologically Sorted Source Nodes: [features_21_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_21_conv_8 => add_118, mul_197, mul_198, sub_50
# Graph fragment:
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_401), kwargs = {})
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_197, %unsqueeze_405), kwargs = {})
#   %add_118 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_198, %unsqueeze_407), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_31', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 176)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uh/cuhlg6oul2hug5v54iht7wanpcyhjozttvpq27ab7pbavjwqxbgc.py
# Topologically Sorted Source Nodes: [features_22_conv_1, sigmoid_69, mul_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_22_conv_1 => add_120, mul_200, mul_201, sub_51
#   mul_47 => mul_202
#   sigmoid_69 => sigmoid_46
# Graph fragment:
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_409), kwargs = {})
#   %mul_200 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_200, %unsqueeze_413), kwargs = {})
#   %add_120 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_201, %unsqueeze_415), kwargs = {})
#   %sigmoid_46 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_120,), kwargs = {})
#   %mul_202 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_120, %sigmoid_46), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1056)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hw/chwztzxtavdbhhpxxeqfauhvw5citmcosdrzw5y2lcg7gvxxywal.py
# Topologically Sorted Source Nodes: [features_22_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_22_conv_4 => add_122, mul_204, mul_205, sub_52
# Graph fragment:
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_417), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_205 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_204, %unsqueeze_421), kwargs = {})
#   %add_122 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_205, %unsqueeze_423), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1056)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u2/cu26edmvu2myd5fjn5zabheno3j35ksem665wd3kanw4gaxkwcrq.py
# Topologically Sorted Source Nodes: [sigmoid_71, mul_48, features_22_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_22_conv_6_avg_pool => mean_8
#   mul_48 => mul_206
#   sigmoid_71 => sigmoid_47
# Graph fragment:
#   %sigmoid_47 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_206 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_47), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_206, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_34 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_34(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4224
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 1056)
    x1 = xindex // 1056
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1056*r2 + 16896*x1), xmask, other=0.0)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 16.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wq/cwqj6j3yt7knvxe2ea564zsuetdthj6gwjpoz5gytzyqno76hl6b.py
# Topologically Sorted Source Nodes: [sigmoid_73, mul_49], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_49 => mul_207
#   sigmoid_73 => sigmoid_48
# Graph fragment:
#   %sigmoid_48 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_16,), kwargs = {})
#   %mul_207 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_16, %sigmoid_48), kwargs = {})
triton_poi_fused_mul_sigmoid_35 = async_compile.triton('triton_poi_fused_mul_sigmoid_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cs/ccsjzsp2zr3pwnmlye623kh4iunuo6c544igx6dygtxmkmy7u4bi.py
# Topologically Sorted Source Nodes: [sigmoid_71, mul_48, mul_50], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_48 => mul_206
#   mul_50 => mul_208
#   sigmoid_71 => sigmoid_47
# Graph fragment:
#   %sigmoid_47 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_122,), kwargs = {})
#   %mul_206 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_122, %sigmoid_47), kwargs = {})
#   %mul_208 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %view_17), kwargs = {})
triton_poi_fused_mul_sigmoid_36 = async_compile.triton('triton_poi_fused_mul_sigmoid_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 67584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 1056)
    x2 = xindex // 16896
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 1056*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/up/cupezzwuczn2koe22oculiob2qucv5kutxs6ytbhuto5h2j3uids.py
# Topologically Sorted Source Nodes: [features_22_conv_8, add_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_18 => add_125
#   features_22_conv_8 => add_124, mul_210, mul_211, sub_53
# Graph fragment:
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_425), kwargs = {})
#   %mul_210 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_211 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_210, %unsqueeze_429), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_211, %unsqueeze_431), kwargs = {})
#   %add_125 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_118, %add_124), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_37(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 176)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qd/cqdmpamq5pnu2hotf2fkurmwn3mu5tq3nnyo6myplldyaemo4trb.py
# Topologically Sorted Source Nodes: [features_35_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_35_conv_4 => add_213, mul_373, mul_374, sub_91
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_729), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_373, %unsqueeze_733), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %unsqueeze_735), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1056)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wl/cwletqkcwtri4b3hzb2vtcwogthmnvd5zq6s55hbgsr34q6wcxss.py
# Topologically Sorted Source Nodes: [sigmoid_162, mul_100, features_35_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_35_conv_6_avg_pool => mean_21
#   mul_100 => mul_375
#   sigmoid_162 => sigmoid_99
# Graph fragment:
#   %sigmoid_99 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_375 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_99), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_375, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_39 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1056)
    x1 = xindex // 1056
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4224*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (1056 + x0 + 4224*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (2112 + x0 + 4224*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (3168 + x0 + 4224*x1), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hg/chgjzud7ebhvs5cffpbpxnnhjjnxbpnmynmfp42c65skza5xtrci.py
# Topologically Sorted Source Nodes: [sigmoid_162, mul_100, mul_102], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_100 => mul_375
#   mul_102 => mul_377
#   sigmoid_162 => sigmoid_99
# Graph fragment:
#   %sigmoid_99 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_375 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_99), kwargs = {})
#   %mul_377 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_375, %view_43), kwargs = {})
triton_poi_fused_mul_sigmoid_40 = async_compile.triton('triton_poi_fused_mul_sigmoid_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_40(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 1056)
    x2 = xindex // 4224
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 1056*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dh/cdhoz4ji743r5sqrqt5eq62lm37yg5w7xt35nmscead5pzha6zq2.py
# Topologically Sorted Source Nodes: [features_35_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_35_conv_8 => add_215, mul_379, mul_380, sub_92
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_737), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_741), kwargs = {})
#   %add_215 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_743), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 304)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pn/cpny6ihj5moo5q3jzo32i4ych7ewigs3t2zu52gycgbzrrgopipz.py
# Topologically Sorted Source Nodes: [features_36_conv_1, sigmoid_167, mul_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_36_conv_1 => add_217, mul_382, mul_383, sub_93
#   mul_103 => mul_384
#   sigmoid_167 => sigmoid_102
# Graph fragment:
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_745), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_749), kwargs = {})
#   %add_217 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_751), kwargs = {})
#   %sigmoid_102 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_217,), kwargs = {})
#   %mul_384 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_217, %sigmoid_102), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1824)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xw/cxwtkozrixrsdwfv6mogifnoegzsqpplp3j2cja2vc67mtdwysyk.py
# Topologically Sorted Source Nodes: [features_36_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_36_conv_4 => add_219, mul_386, mul_387, sub_94
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %mul_386 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_386, %unsqueeze_757), kwargs = {})
#   %add_219 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_387, %unsqueeze_759), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_43', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_43(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1824)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yo/cyoivvekxnlcbq2klchrwlnrgo5rytthoqj5h2x3ocxfb2zdyim7.py
# Topologically Sorted Source Nodes: [sigmoid_169, mul_104, features_36_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_36_conv_6_avg_pool => mean_22
#   mul_104 => mul_388
#   sigmoid_169 => sigmoid_103
# Graph fragment:
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_219,), kwargs = {})
#   %mul_388 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_219, %sigmoid_103), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_388, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_44 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_44(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1824)
    x1 = xindex // 1824
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 7296*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (1824 + x0 + 7296*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (3648 + x0 + 7296*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (5472 + x0 + 7296*x1), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hy/chy7oz7njj3edijj7scibiwcpowvcgj4pndzwajeam4kogwpqv2u.py
# Topologically Sorted Source Nodes: [sigmoid_171, mul_105], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_105 => mul_389
#   sigmoid_171 => sigmoid_104
# Graph fragment:
#   %sigmoid_104 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_44,), kwargs = {})
#   %mul_389 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_44, %sigmoid_104), kwargs = {})
triton_poi_fused_mul_sigmoid_45 = async_compile.triton('triton_poi_fused_mul_sigmoid_45', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/62/c62htlojdn44gq5xrgqw2tszky2bxsfhffo7ranw54zzlqbkuxb6.py
# Topologically Sorted Source Nodes: [sigmoid_169, mul_104, mul_106], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_104 => mul_388
#   mul_106 => mul_390
#   sigmoid_169 => sigmoid_103
# Graph fragment:
#   %sigmoid_103 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_219,), kwargs = {})
#   %mul_388 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_219, %sigmoid_103), kwargs = {})
#   %mul_390 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_388, %view_45), kwargs = {})
triton_poi_fused_mul_sigmoid_46 = async_compile.triton('triton_poi_fused_mul_sigmoid_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_46(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 1824)
    x2 = xindex // 7296
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 1824*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eu/ceuwba6w3s2f2atntwzrx6hj4ylbpjk32lh5ekypfigvcuec7ahc.py
# Topologically Sorted Source Nodes: [features_36_conv_8, add_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_31 => add_222
#   features_36_conv_8 => add_221, mul_392, mul_393, sub_95
# Graph fragment:
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_761), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_392, %unsqueeze_765), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_393, %unsqueeze_767), kwargs = {})
#   %add_222 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_215, %add_221), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 304)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pl/cpliwuuwt5samvnfm65umv2rj7w6yeiehperbf3ntsy4frcvgtke.py
# Topologically Sorted Source Nodes: [features_53_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_53_conv_8 => add_340, mul_613, mul_614, sub_146
# Graph fragment:
#   %sub_146 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_146, %unsqueeze_1169), kwargs = {})
#   %mul_613 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_146, %unsqueeze_1171), kwargs = {})
#   %mul_614 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_613, %unsqueeze_1173), kwargs = {})
#   %add_340 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_614, %unsqueeze_1175), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_48', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/cs/ccspmezizrsh6uioxv56zo5jtpestzjo3yqltkqzt3cu62fhi7py.py
# Topologically Sorted Source Nodes: [features_54_conv_1, sigmoid_293, mul_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_54_conv_1 => add_342, mul_616, mul_617, sub_147
#   mul_175 => mul_618
#   sigmoid_293 => sigmoid_174
# Graph fragment:
#   %sub_147 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_147, %unsqueeze_1177), kwargs = {})
#   %mul_616 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_147, %unsqueeze_1179), kwargs = {})
#   %mul_617 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_616, %unsqueeze_1181), kwargs = {})
#   %add_342 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_617, %unsqueeze_1183), kwargs = {})
#   %sigmoid_174 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_342,), kwargs = {})
#   %mul_618 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_342, %sigmoid_174), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ck/cckrxq4vrz7izjqttx2wscnju2ksizqywe7lni3ksajwtzggq3e6.py
# Topologically Sorted Source Nodes: [features_54_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_54_conv_4 => add_344, mul_620, mul_621, sub_148
# Graph fragment:
#   %sub_148 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_148, %unsqueeze_1185), kwargs = {})
#   %mul_620 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_148, %unsqueeze_1187), kwargs = {})
#   %mul_621 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_620, %unsqueeze_1189), kwargs = {})
#   %add_344 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_621, %unsqueeze_1191), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3072)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/4p/c4pjqhfjxpry7xd6ad62dw4yritpslrp3nqdtikgl3vagbueeur4.py
# Topologically Sorted Source Nodes: [sigmoid_295, mul_176, features_54_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_54_conv_6_avg_pool => mean_40
#   mul_176 => mul_622
#   sigmoid_295 => sigmoid_175
# Graph fragment:
#   %sigmoid_175 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_344,), kwargs = {})
#   %mul_622 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_344, %sigmoid_175), kwargs = {})
#   %mean_40 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_622, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_51 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_51(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12288*x1), None)
    tmp3 = tl.load(in_ptr0 + (3072 + x0 + 12288*x1), None)
    tmp7 = tl.load(in_ptr0 + (6144 + x0 + 12288*x1), None)
    tmp11 = tl.load(in_ptr0 + (9216 + x0 + 12288*x1), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + (x2), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/zm/czmbgpb5jk5gcqip4bw357u5szyz322cjrfyw4kxvfmcxgatar4i.py
# Topologically Sorted Source Nodes: [sigmoid_297, mul_177], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_177 => mul_623
#   sigmoid_297 => sigmoid_176
# Graph fragment:
#   %sigmoid_176 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_80,), kwargs = {})
#   %mul_623 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_80, %sigmoid_176), kwargs = {})
triton_poi_fused_mul_sigmoid_52 = async_compile.triton('triton_poi_fused_mul_sigmoid_52', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_52', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_52(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/en/cenecf7hdgfwnb5tjukgnyu36p5aaj3nkiwxpwtf2ujytuwwqwvk.py
# Topologically Sorted Source Nodes: [sigmoid_295, mul_176, mul_178], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_176 => mul_622
#   mul_178 => mul_624
#   sigmoid_295 => sigmoid_175
# Graph fragment:
#   %sigmoid_175 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_344,), kwargs = {})
#   %mul_622 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_344, %sigmoid_175), kwargs = {})
#   %mul_624 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_622, %view_81), kwargs = {})
triton_poi_fused_mul_sigmoid_53 = async_compile.triton('triton_poi_fused_mul_sigmoid_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_53(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 3072)
    x2 = xindex // 12288
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 3072*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/7z/c7zmugunqkbxqho4tx42dnxu44oh5yxnhsk4egxm5hct75ofkne5.py
# Topologically Sorted Source Nodes: [features_54_conv_8, add_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_48 => add_347
#   features_54_conv_8 => add_346, mul_626, mul_627, sub_149
# Graph fragment:
#   %sub_149 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_149, %unsqueeze_1193), kwargs = {})
#   %mul_626 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_149, %unsqueeze_1195), kwargs = {})
#   %mul_627 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_626, %unsqueeze_1197), kwargs = {})
#   %add_346 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_627, %unsqueeze_1199), kwargs = {})
#   %add_347 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_340, %add_346), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_54', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_54(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/wp/cwpktnmtpjubolzhszmlsid36i62lccoc6ukakmod636sssjudfd.py
# Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv_1 => add_370, mul_668, mul_669, sub_159
# Graph fragment:
#   %sub_159 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_159, %unsqueeze_1273), kwargs = {})
#   %mul_668 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_159, %unsqueeze_1275), kwargs = {})
#   %mul_669 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_668, %unsqueeze_1277), kwargs = {})
#   %add_370 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_669, %unsqueeze_1279), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_55', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1792)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/vv/cvvsjob4dlzypuuf47us6slnpysitvg6iyqtbhlv3zgbikcufpel.py
# Topologically Sorted Source Nodes: [sigmoid_321, mul_191, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   avgpool => mean_44
#   mul_191 => mul_670
#   sigmoid_321 => sigmoid_190
# Graph fragment:
#   %sigmoid_190 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_370,), kwargs = {})
#   %mul_670 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_370, %sigmoid_190), kwargs = {})
#   %mean_44 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_670, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_56 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_56', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_56(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1792)
    x1 = xindex // 1792
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 7168*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (1792 + x0 + 7168*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (3584 + x0 + 7168*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (5376 + x0 + 7168*x1), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tl.sigmoid(tmp7)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979 = args
    args.clear()
    assert_size_stride(primals_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (24, ), (1, ))
    assert_size_stride(primals_4, (24, ), (1, ))
    assert_size_stride(primals_5, (24, ), (1, ))
    assert_size_stride(primals_6, (24, ), (1, ))
    assert_size_stride(primals_7, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_8, (24, ), (1, ))
    assert_size_stride(primals_9, (24, ), (1, ))
    assert_size_stride(primals_10, (24, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (24, ), (1, ))
    assert_size_stride(primals_15, (24, ), (1, ))
    assert_size_stride(primals_16, (24, ), (1, ))
    assert_size_stride(primals_17, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_18, (24, ), (1, ))
    assert_size_stride(primals_19, (24, ), (1, ))
    assert_size_stride(primals_20, (24, ), (1, ))
    assert_size_stride(primals_21, (24, ), (1, ))
    assert_size_stride(primals_22, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_23, (24, ), (1, ))
    assert_size_stride(primals_24, (24, ), (1, ))
    assert_size_stride(primals_25, (24, ), (1, ))
    assert_size_stride(primals_26, (24, ), (1, ))
    assert_size_stride(primals_27, (24, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_28, (24, ), (1, ))
    assert_size_stride(primals_29, (24, ), (1, ))
    assert_size_stride(primals_30, (24, ), (1, ))
    assert_size_stride(primals_31, (24, ), (1, ))
    assert_size_stride(primals_32, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_33, (24, ), (1, ))
    assert_size_stride(primals_34, (24, ), (1, ))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_36, (24, ), (1, ))
    assert_size_stride(primals_37, (96, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_38, (96, ), (1, ))
    assert_size_stride(primals_39, (96, ), (1, ))
    assert_size_stride(primals_40, (96, ), (1, ))
    assert_size_stride(primals_41, (96, ), (1, ))
    assert_size_stride(primals_42, (48, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_43, (48, ), (1, ))
    assert_size_stride(primals_44, (48, ), (1, ))
    assert_size_stride(primals_45, (48, ), (1, ))
    assert_size_stride(primals_46, (48, ), (1, ))
    assert_size_stride(primals_47, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_53, (48, ), (1, ))
    assert_size_stride(primals_54, (48, ), (1, ))
    assert_size_stride(primals_55, (48, ), (1, ))
    assert_size_stride(primals_56, (48, ), (1, ))
    assert_size_stride(primals_57, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_58, (192, ), (1, ))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_63, (48, ), (1, ))
    assert_size_stride(primals_64, (48, ), (1, ))
    assert_size_stride(primals_65, (48, ), (1, ))
    assert_size_stride(primals_66, (48, ), (1, ))
    assert_size_stride(primals_67, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_68, (192, ), (1, ))
    assert_size_stride(primals_69, (192, ), (1, ))
    assert_size_stride(primals_70, (192, ), (1, ))
    assert_size_stride(primals_71, (192, ), (1, ))
    assert_size_stride(primals_72, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_73, (48, ), (1, ))
    assert_size_stride(primals_74, (48, ), (1, ))
    assert_size_stride(primals_75, (48, ), (1, ))
    assert_size_stride(primals_76, (48, ), (1, ))
    assert_size_stride(primals_77, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_78, (192, ), (1, ))
    assert_size_stride(primals_79, (192, ), (1, ))
    assert_size_stride(primals_80, (192, ), (1, ))
    assert_size_stride(primals_81, (192, ), (1, ))
    assert_size_stride(primals_82, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_83, (48, ), (1, ))
    assert_size_stride(primals_84, (48, ), (1, ))
    assert_size_stride(primals_85, (48, ), (1, ))
    assert_size_stride(primals_86, (48, ), (1, ))
    assert_size_stride(primals_87, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_90, (192, ), (1, ))
    assert_size_stride(primals_91, (192, ), (1, ))
    assert_size_stride(primals_92, (80, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_93, (80, ), (1, ))
    assert_size_stride(primals_94, (80, ), (1, ))
    assert_size_stride(primals_95, (80, ), (1, ))
    assert_size_stride(primals_96, (80, ), (1, ))
    assert_size_stride(primals_97, (320, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_98, (320, ), (1, ))
    assert_size_stride(primals_99, (320, ), (1, ))
    assert_size_stride(primals_100, (320, ), (1, ))
    assert_size_stride(primals_101, (320, ), (1, ))
    assert_size_stride(primals_102, (80, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_103, (80, ), (1, ))
    assert_size_stride(primals_104, (80, ), (1, ))
    assert_size_stride(primals_105, (80, ), (1, ))
    assert_size_stride(primals_106, (80, ), (1, ))
    assert_size_stride(primals_107, (320, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_108, (320, ), (1, ))
    assert_size_stride(primals_109, (320, ), (1, ))
    assert_size_stride(primals_110, (320, ), (1, ))
    assert_size_stride(primals_111, (320, ), (1, ))
    assert_size_stride(primals_112, (80, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_113, (80, ), (1, ))
    assert_size_stride(primals_114, (80, ), (1, ))
    assert_size_stride(primals_115, (80, ), (1, ))
    assert_size_stride(primals_116, (80, ), (1, ))
    assert_size_stride(primals_117, (320, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_118, (320, ), (1, ))
    assert_size_stride(primals_119, (320, ), (1, ))
    assert_size_stride(primals_120, (320, ), (1, ))
    assert_size_stride(primals_121, (320, ), (1, ))
    assert_size_stride(primals_122, (80, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_123, (80, ), (1, ))
    assert_size_stride(primals_124, (80, ), (1, ))
    assert_size_stride(primals_125, (80, ), (1, ))
    assert_size_stride(primals_126, (80, ), (1, ))
    assert_size_stride(primals_127, (320, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(primals_128, (320, ), (1, ))
    assert_size_stride(primals_129, (320, ), (1, ))
    assert_size_stride(primals_130, (320, ), (1, ))
    assert_size_stride(primals_131, (320, ), (1, ))
    assert_size_stride(primals_132, (80, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_133, (80, ), (1, ))
    assert_size_stride(primals_134, (80, ), (1, ))
    assert_size_stride(primals_135, (80, ), (1, ))
    assert_size_stride(primals_136, (80, ), (1, ))
    assert_size_stride(primals_137, (320, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(primals_138, (320, ), (1, ))
    assert_size_stride(primals_139, (320, ), (1, ))
    assert_size_stride(primals_140, (320, ), (1, ))
    assert_size_stride(primals_141, (320, ), (1, ))
    assert_size_stride(primals_142, (320, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (320, ), (1, ))
    assert_size_stride(primals_144, (320, ), (1, ))
    assert_size_stride(primals_145, (320, ), (1, ))
    assert_size_stride(primals_146, (320, ), (1, ))
    assert_size_stride(primals_147, (24, 320), (320, 1))
    assert_size_stride(primals_148, (24, ), (1, ))
    assert_size_stride(primals_149, (320, 24), (24, 1))
    assert_size_stride(primals_150, (320, ), (1, ))
    assert_size_stride(primals_151, (160, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_152, (160, ), (1, ))
    assert_size_stride(primals_153, (160, ), (1, ))
    assert_size_stride(primals_154, (160, ), (1, ))
    assert_size_stride(primals_155, (160, ), (1, ))
    assert_size_stride(primals_156, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_157, (640, ), (1, ))
    assert_size_stride(primals_158, (640, ), (1, ))
    assert_size_stride(primals_159, (640, ), (1, ))
    assert_size_stride(primals_160, (640, ), (1, ))
    assert_size_stride(primals_161, (640, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_162, (640, ), (1, ))
    assert_size_stride(primals_163, (640, ), (1, ))
    assert_size_stride(primals_164, (640, ), (1, ))
    assert_size_stride(primals_165, (640, ), (1, ))
    assert_size_stride(primals_166, (40, 640), (640, 1))
    assert_size_stride(primals_167, (40, ), (1, ))
    assert_size_stride(primals_168, (640, 40), (40, 1))
    assert_size_stride(primals_169, (640, ), (1, ))
    assert_size_stride(primals_170, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_171, (160, ), (1, ))
    assert_size_stride(primals_172, (160, ), (1, ))
    assert_size_stride(primals_173, (160, ), (1, ))
    assert_size_stride(primals_174, (160, ), (1, ))
    assert_size_stride(primals_175, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_176, (640, ), (1, ))
    assert_size_stride(primals_177, (640, ), (1, ))
    assert_size_stride(primals_178, (640, ), (1, ))
    assert_size_stride(primals_179, (640, ), (1, ))
    assert_size_stride(primals_180, (640, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_181, (640, ), (1, ))
    assert_size_stride(primals_182, (640, ), (1, ))
    assert_size_stride(primals_183, (640, ), (1, ))
    assert_size_stride(primals_184, (640, ), (1, ))
    assert_size_stride(primals_185, (40, 640), (640, 1))
    assert_size_stride(primals_186, (40, ), (1, ))
    assert_size_stride(primals_187, (640, 40), (40, 1))
    assert_size_stride(primals_188, (640, ), (1, ))
    assert_size_stride(primals_189, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_190, (160, ), (1, ))
    assert_size_stride(primals_191, (160, ), (1, ))
    assert_size_stride(primals_192, (160, ), (1, ))
    assert_size_stride(primals_193, (160, ), (1, ))
    assert_size_stride(primals_194, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_195, (640, ), (1, ))
    assert_size_stride(primals_196, (640, ), (1, ))
    assert_size_stride(primals_197, (640, ), (1, ))
    assert_size_stride(primals_198, (640, ), (1, ))
    assert_size_stride(primals_199, (640, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_200, (640, ), (1, ))
    assert_size_stride(primals_201, (640, ), (1, ))
    assert_size_stride(primals_202, (640, ), (1, ))
    assert_size_stride(primals_203, (640, ), (1, ))
    assert_size_stride(primals_204, (40, 640), (640, 1))
    assert_size_stride(primals_205, (40, ), (1, ))
    assert_size_stride(primals_206, (640, 40), (40, 1))
    assert_size_stride(primals_207, (640, ), (1, ))
    assert_size_stride(primals_208, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_209, (160, ), (1, ))
    assert_size_stride(primals_210, (160, ), (1, ))
    assert_size_stride(primals_211, (160, ), (1, ))
    assert_size_stride(primals_212, (160, ), (1, ))
    assert_size_stride(primals_213, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_214, (640, ), (1, ))
    assert_size_stride(primals_215, (640, ), (1, ))
    assert_size_stride(primals_216, (640, ), (1, ))
    assert_size_stride(primals_217, (640, ), (1, ))
    assert_size_stride(primals_218, (640, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_219, (640, ), (1, ))
    assert_size_stride(primals_220, (640, ), (1, ))
    assert_size_stride(primals_221, (640, ), (1, ))
    assert_size_stride(primals_222, (640, ), (1, ))
    assert_size_stride(primals_223, (40, 640), (640, 1))
    assert_size_stride(primals_224, (40, ), (1, ))
    assert_size_stride(primals_225, (640, 40), (40, 1))
    assert_size_stride(primals_226, (640, ), (1, ))
    assert_size_stride(primals_227, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_228, (160, ), (1, ))
    assert_size_stride(primals_229, (160, ), (1, ))
    assert_size_stride(primals_230, (160, ), (1, ))
    assert_size_stride(primals_231, (160, ), (1, ))
    assert_size_stride(primals_232, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_233, (640, ), (1, ))
    assert_size_stride(primals_234, (640, ), (1, ))
    assert_size_stride(primals_235, (640, ), (1, ))
    assert_size_stride(primals_236, (640, ), (1, ))
    assert_size_stride(primals_237, (640, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_238, (640, ), (1, ))
    assert_size_stride(primals_239, (640, ), (1, ))
    assert_size_stride(primals_240, (640, ), (1, ))
    assert_size_stride(primals_241, (640, ), (1, ))
    assert_size_stride(primals_242, (40, 640), (640, 1))
    assert_size_stride(primals_243, (40, ), (1, ))
    assert_size_stride(primals_244, (640, 40), (40, 1))
    assert_size_stride(primals_245, (640, ), (1, ))
    assert_size_stride(primals_246, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_247, (160, ), (1, ))
    assert_size_stride(primals_248, (160, ), (1, ))
    assert_size_stride(primals_249, (160, ), (1, ))
    assert_size_stride(primals_250, (160, ), (1, ))
    assert_size_stride(primals_251, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_252, (640, ), (1, ))
    assert_size_stride(primals_253, (640, ), (1, ))
    assert_size_stride(primals_254, (640, ), (1, ))
    assert_size_stride(primals_255, (640, ), (1, ))
    assert_size_stride(primals_256, (640, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_257, (640, ), (1, ))
    assert_size_stride(primals_258, (640, ), (1, ))
    assert_size_stride(primals_259, (640, ), (1, ))
    assert_size_stride(primals_260, (640, ), (1, ))
    assert_size_stride(primals_261, (40, 640), (640, 1))
    assert_size_stride(primals_262, (40, ), (1, ))
    assert_size_stride(primals_263, (640, 40), (40, 1))
    assert_size_stride(primals_264, (640, ), (1, ))
    assert_size_stride(primals_265, (160, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_266, (160, ), (1, ))
    assert_size_stride(primals_267, (160, ), (1, ))
    assert_size_stride(primals_268, (160, ), (1, ))
    assert_size_stride(primals_269, (160, ), (1, ))
    assert_size_stride(primals_270, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_271, (960, ), (1, ))
    assert_size_stride(primals_272, (960, ), (1, ))
    assert_size_stride(primals_273, (960, ), (1, ))
    assert_size_stride(primals_274, (960, ), (1, ))
    assert_size_stride(primals_275, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_276, (960, ), (1, ))
    assert_size_stride(primals_277, (960, ), (1, ))
    assert_size_stride(primals_278, (960, ), (1, ))
    assert_size_stride(primals_279, (960, ), (1, ))
    assert_size_stride(primals_280, (40, 960), (960, 1))
    assert_size_stride(primals_281, (40, ), (1, ))
    assert_size_stride(primals_282, (960, 40), (40, 1))
    assert_size_stride(primals_283, (960, ), (1, ))
    assert_size_stride(primals_284, (176, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_285, (176, ), (1, ))
    assert_size_stride(primals_286, (176, ), (1, ))
    assert_size_stride(primals_287, (176, ), (1, ))
    assert_size_stride(primals_288, (176, ), (1, ))
    assert_size_stride(primals_289, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_290, (1056, ), (1, ))
    assert_size_stride(primals_291, (1056, ), (1, ))
    assert_size_stride(primals_292, (1056, ), (1, ))
    assert_size_stride(primals_293, (1056, ), (1, ))
    assert_size_stride(primals_294, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_295, (1056, ), (1, ))
    assert_size_stride(primals_296, (1056, ), (1, ))
    assert_size_stride(primals_297, (1056, ), (1, ))
    assert_size_stride(primals_298, (1056, ), (1, ))
    assert_size_stride(primals_299, (48, 1056), (1056, 1))
    assert_size_stride(primals_300, (48, ), (1, ))
    assert_size_stride(primals_301, (1056, 48), (48, 1))
    assert_size_stride(primals_302, (1056, ), (1, ))
    assert_size_stride(primals_303, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_304, (176, ), (1, ))
    assert_size_stride(primals_305, (176, ), (1, ))
    assert_size_stride(primals_306, (176, ), (1, ))
    assert_size_stride(primals_307, (176, ), (1, ))
    assert_size_stride(primals_308, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_309, (1056, ), (1, ))
    assert_size_stride(primals_310, (1056, ), (1, ))
    assert_size_stride(primals_311, (1056, ), (1, ))
    assert_size_stride(primals_312, (1056, ), (1, ))
    assert_size_stride(primals_313, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_314, (1056, ), (1, ))
    assert_size_stride(primals_315, (1056, ), (1, ))
    assert_size_stride(primals_316, (1056, ), (1, ))
    assert_size_stride(primals_317, (1056, ), (1, ))
    assert_size_stride(primals_318, (48, 1056), (1056, 1))
    assert_size_stride(primals_319, (48, ), (1, ))
    assert_size_stride(primals_320, (1056, 48), (48, 1))
    assert_size_stride(primals_321, (1056, ), (1, ))
    assert_size_stride(primals_322, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_323, (176, ), (1, ))
    assert_size_stride(primals_324, (176, ), (1, ))
    assert_size_stride(primals_325, (176, ), (1, ))
    assert_size_stride(primals_326, (176, ), (1, ))
    assert_size_stride(primals_327, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_328, (1056, ), (1, ))
    assert_size_stride(primals_329, (1056, ), (1, ))
    assert_size_stride(primals_330, (1056, ), (1, ))
    assert_size_stride(primals_331, (1056, ), (1, ))
    assert_size_stride(primals_332, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_333, (1056, ), (1, ))
    assert_size_stride(primals_334, (1056, ), (1, ))
    assert_size_stride(primals_335, (1056, ), (1, ))
    assert_size_stride(primals_336, (1056, ), (1, ))
    assert_size_stride(primals_337, (48, 1056), (1056, 1))
    assert_size_stride(primals_338, (48, ), (1, ))
    assert_size_stride(primals_339, (1056, 48), (48, 1))
    assert_size_stride(primals_340, (1056, ), (1, ))
    assert_size_stride(primals_341, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_342, (176, ), (1, ))
    assert_size_stride(primals_343, (176, ), (1, ))
    assert_size_stride(primals_344, (176, ), (1, ))
    assert_size_stride(primals_345, (176, ), (1, ))
    assert_size_stride(primals_346, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_347, (1056, ), (1, ))
    assert_size_stride(primals_348, (1056, ), (1, ))
    assert_size_stride(primals_349, (1056, ), (1, ))
    assert_size_stride(primals_350, (1056, ), (1, ))
    assert_size_stride(primals_351, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_352, (1056, ), (1, ))
    assert_size_stride(primals_353, (1056, ), (1, ))
    assert_size_stride(primals_354, (1056, ), (1, ))
    assert_size_stride(primals_355, (1056, ), (1, ))
    assert_size_stride(primals_356, (48, 1056), (1056, 1))
    assert_size_stride(primals_357, (48, ), (1, ))
    assert_size_stride(primals_358, (1056, 48), (48, 1))
    assert_size_stride(primals_359, (1056, ), (1, ))
    assert_size_stride(primals_360, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_361, (176, ), (1, ))
    assert_size_stride(primals_362, (176, ), (1, ))
    assert_size_stride(primals_363, (176, ), (1, ))
    assert_size_stride(primals_364, (176, ), (1, ))
    assert_size_stride(primals_365, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_366, (1056, ), (1, ))
    assert_size_stride(primals_367, (1056, ), (1, ))
    assert_size_stride(primals_368, (1056, ), (1, ))
    assert_size_stride(primals_369, (1056, ), (1, ))
    assert_size_stride(primals_370, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_371, (1056, ), (1, ))
    assert_size_stride(primals_372, (1056, ), (1, ))
    assert_size_stride(primals_373, (1056, ), (1, ))
    assert_size_stride(primals_374, (1056, ), (1, ))
    assert_size_stride(primals_375, (48, 1056), (1056, 1))
    assert_size_stride(primals_376, (48, ), (1, ))
    assert_size_stride(primals_377, (1056, 48), (48, 1))
    assert_size_stride(primals_378, (1056, ), (1, ))
    assert_size_stride(primals_379, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_380, (176, ), (1, ))
    assert_size_stride(primals_381, (176, ), (1, ))
    assert_size_stride(primals_382, (176, ), (1, ))
    assert_size_stride(primals_383, (176, ), (1, ))
    assert_size_stride(primals_384, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_385, (1056, ), (1, ))
    assert_size_stride(primals_386, (1056, ), (1, ))
    assert_size_stride(primals_387, (1056, ), (1, ))
    assert_size_stride(primals_388, (1056, ), (1, ))
    assert_size_stride(primals_389, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_390, (1056, ), (1, ))
    assert_size_stride(primals_391, (1056, ), (1, ))
    assert_size_stride(primals_392, (1056, ), (1, ))
    assert_size_stride(primals_393, (1056, ), (1, ))
    assert_size_stride(primals_394, (48, 1056), (1056, 1))
    assert_size_stride(primals_395, (48, ), (1, ))
    assert_size_stride(primals_396, (1056, 48), (48, 1))
    assert_size_stride(primals_397, (1056, ), (1, ))
    assert_size_stride(primals_398, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_399, (176, ), (1, ))
    assert_size_stride(primals_400, (176, ), (1, ))
    assert_size_stride(primals_401, (176, ), (1, ))
    assert_size_stride(primals_402, (176, ), (1, ))
    assert_size_stride(primals_403, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_404, (1056, ), (1, ))
    assert_size_stride(primals_405, (1056, ), (1, ))
    assert_size_stride(primals_406, (1056, ), (1, ))
    assert_size_stride(primals_407, (1056, ), (1, ))
    assert_size_stride(primals_408, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_409, (1056, ), (1, ))
    assert_size_stride(primals_410, (1056, ), (1, ))
    assert_size_stride(primals_411, (1056, ), (1, ))
    assert_size_stride(primals_412, (1056, ), (1, ))
    assert_size_stride(primals_413, (48, 1056), (1056, 1))
    assert_size_stride(primals_414, (48, ), (1, ))
    assert_size_stride(primals_415, (1056, 48), (48, 1))
    assert_size_stride(primals_416, (1056, ), (1, ))
    assert_size_stride(primals_417, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_418, (176, ), (1, ))
    assert_size_stride(primals_419, (176, ), (1, ))
    assert_size_stride(primals_420, (176, ), (1, ))
    assert_size_stride(primals_421, (176, ), (1, ))
    assert_size_stride(primals_422, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_423, (1056, ), (1, ))
    assert_size_stride(primals_424, (1056, ), (1, ))
    assert_size_stride(primals_425, (1056, ), (1, ))
    assert_size_stride(primals_426, (1056, ), (1, ))
    assert_size_stride(primals_427, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_428, (1056, ), (1, ))
    assert_size_stride(primals_429, (1056, ), (1, ))
    assert_size_stride(primals_430, (1056, ), (1, ))
    assert_size_stride(primals_431, (1056, ), (1, ))
    assert_size_stride(primals_432, (48, 1056), (1056, 1))
    assert_size_stride(primals_433, (48, ), (1, ))
    assert_size_stride(primals_434, (1056, 48), (48, 1))
    assert_size_stride(primals_435, (1056, ), (1, ))
    assert_size_stride(primals_436, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_437, (176, ), (1, ))
    assert_size_stride(primals_438, (176, ), (1, ))
    assert_size_stride(primals_439, (176, ), (1, ))
    assert_size_stride(primals_440, (176, ), (1, ))
    assert_size_stride(primals_441, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_442, (1056, ), (1, ))
    assert_size_stride(primals_443, (1056, ), (1, ))
    assert_size_stride(primals_444, (1056, ), (1, ))
    assert_size_stride(primals_445, (1056, ), (1, ))
    assert_size_stride(primals_446, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_447, (1056, ), (1, ))
    assert_size_stride(primals_448, (1056, ), (1, ))
    assert_size_stride(primals_449, (1056, ), (1, ))
    assert_size_stride(primals_450, (1056, ), (1, ))
    assert_size_stride(primals_451, (48, 1056), (1056, 1))
    assert_size_stride(primals_452, (48, ), (1, ))
    assert_size_stride(primals_453, (1056, 48), (48, 1))
    assert_size_stride(primals_454, (1056, ), (1, ))
    assert_size_stride(primals_455, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_456, (176, ), (1, ))
    assert_size_stride(primals_457, (176, ), (1, ))
    assert_size_stride(primals_458, (176, ), (1, ))
    assert_size_stride(primals_459, (176, ), (1, ))
    assert_size_stride(primals_460, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_461, (1056, ), (1, ))
    assert_size_stride(primals_462, (1056, ), (1, ))
    assert_size_stride(primals_463, (1056, ), (1, ))
    assert_size_stride(primals_464, (1056, ), (1, ))
    assert_size_stride(primals_465, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_466, (1056, ), (1, ))
    assert_size_stride(primals_467, (1056, ), (1, ))
    assert_size_stride(primals_468, (1056, ), (1, ))
    assert_size_stride(primals_469, (1056, ), (1, ))
    assert_size_stride(primals_470, (48, 1056), (1056, 1))
    assert_size_stride(primals_471, (48, ), (1, ))
    assert_size_stride(primals_472, (1056, 48), (48, 1))
    assert_size_stride(primals_473, (1056, ), (1, ))
    assert_size_stride(primals_474, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_475, (176, ), (1, ))
    assert_size_stride(primals_476, (176, ), (1, ))
    assert_size_stride(primals_477, (176, ), (1, ))
    assert_size_stride(primals_478, (176, ), (1, ))
    assert_size_stride(primals_479, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_480, (1056, ), (1, ))
    assert_size_stride(primals_481, (1056, ), (1, ))
    assert_size_stride(primals_482, (1056, ), (1, ))
    assert_size_stride(primals_483, (1056, ), (1, ))
    assert_size_stride(primals_484, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_485, (1056, ), (1, ))
    assert_size_stride(primals_486, (1056, ), (1, ))
    assert_size_stride(primals_487, (1056, ), (1, ))
    assert_size_stride(primals_488, (1056, ), (1, ))
    assert_size_stride(primals_489, (48, 1056), (1056, 1))
    assert_size_stride(primals_490, (48, ), (1, ))
    assert_size_stride(primals_491, (1056, 48), (48, 1))
    assert_size_stride(primals_492, (1056, ), (1, ))
    assert_size_stride(primals_493, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_494, (176, ), (1, ))
    assert_size_stride(primals_495, (176, ), (1, ))
    assert_size_stride(primals_496, (176, ), (1, ))
    assert_size_stride(primals_497, (176, ), (1, ))
    assert_size_stride(primals_498, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_499, (1056, ), (1, ))
    assert_size_stride(primals_500, (1056, ), (1, ))
    assert_size_stride(primals_501, (1056, ), (1, ))
    assert_size_stride(primals_502, (1056, ), (1, ))
    assert_size_stride(primals_503, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_504, (1056, ), (1, ))
    assert_size_stride(primals_505, (1056, ), (1, ))
    assert_size_stride(primals_506, (1056, ), (1, ))
    assert_size_stride(primals_507, (1056, ), (1, ))
    assert_size_stride(primals_508, (48, 1056), (1056, 1))
    assert_size_stride(primals_509, (48, ), (1, ))
    assert_size_stride(primals_510, (1056, 48), (48, 1))
    assert_size_stride(primals_511, (1056, ), (1, ))
    assert_size_stride(primals_512, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_513, (176, ), (1, ))
    assert_size_stride(primals_514, (176, ), (1, ))
    assert_size_stride(primals_515, (176, ), (1, ))
    assert_size_stride(primals_516, (176, ), (1, ))
    assert_size_stride(primals_517, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_518, (1056, ), (1, ))
    assert_size_stride(primals_519, (1056, ), (1, ))
    assert_size_stride(primals_520, (1056, ), (1, ))
    assert_size_stride(primals_521, (1056, ), (1, ))
    assert_size_stride(primals_522, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_523, (1056, ), (1, ))
    assert_size_stride(primals_524, (1056, ), (1, ))
    assert_size_stride(primals_525, (1056, ), (1, ))
    assert_size_stride(primals_526, (1056, ), (1, ))
    assert_size_stride(primals_527, (48, 1056), (1056, 1))
    assert_size_stride(primals_528, (48, ), (1, ))
    assert_size_stride(primals_529, (1056, 48), (48, 1))
    assert_size_stride(primals_530, (1056, ), (1, ))
    assert_size_stride(primals_531, (176, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_532, (176, ), (1, ))
    assert_size_stride(primals_533, (176, ), (1, ))
    assert_size_stride(primals_534, (176, ), (1, ))
    assert_size_stride(primals_535, (176, ), (1, ))
    assert_size_stride(primals_536, (1056, 176, 1, 1), (176, 1, 1, 1))
    assert_size_stride(primals_537, (1056, ), (1, ))
    assert_size_stride(primals_538, (1056, ), (1, ))
    assert_size_stride(primals_539, (1056, ), (1, ))
    assert_size_stride(primals_540, (1056, ), (1, ))
    assert_size_stride(primals_541, (1056, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_542, (1056, ), (1, ))
    assert_size_stride(primals_543, (1056, ), (1, ))
    assert_size_stride(primals_544, (1056, ), (1, ))
    assert_size_stride(primals_545, (1056, ), (1, ))
    assert_size_stride(primals_546, (48, 1056), (1056, 1))
    assert_size_stride(primals_547, (48, ), (1, ))
    assert_size_stride(primals_548, (1056, 48), (48, 1))
    assert_size_stride(primals_549, (1056, ), (1, ))
    assert_size_stride(primals_550, (304, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_551, (304, ), (1, ))
    assert_size_stride(primals_552, (304, ), (1, ))
    assert_size_stride(primals_553, (304, ), (1, ))
    assert_size_stride(primals_554, (304, ), (1, ))
    assert_size_stride(primals_555, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_556, (1824, ), (1, ))
    assert_size_stride(primals_557, (1824, ), (1, ))
    assert_size_stride(primals_558, (1824, ), (1, ))
    assert_size_stride(primals_559, (1824, ), (1, ))
    assert_size_stride(primals_560, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_561, (1824, ), (1, ))
    assert_size_stride(primals_562, (1824, ), (1, ))
    assert_size_stride(primals_563, (1824, ), (1, ))
    assert_size_stride(primals_564, (1824, ), (1, ))
    assert_size_stride(primals_565, (80, 1824), (1824, 1))
    assert_size_stride(primals_566, (80, ), (1, ))
    assert_size_stride(primals_567, (1824, 80), (80, 1))
    assert_size_stride(primals_568, (1824, ), (1, ))
    assert_size_stride(primals_569, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_570, (304, ), (1, ))
    assert_size_stride(primals_571, (304, ), (1, ))
    assert_size_stride(primals_572, (304, ), (1, ))
    assert_size_stride(primals_573, (304, ), (1, ))
    assert_size_stride(primals_574, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_575, (1824, ), (1, ))
    assert_size_stride(primals_576, (1824, ), (1, ))
    assert_size_stride(primals_577, (1824, ), (1, ))
    assert_size_stride(primals_578, (1824, ), (1, ))
    assert_size_stride(primals_579, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_580, (1824, ), (1, ))
    assert_size_stride(primals_581, (1824, ), (1, ))
    assert_size_stride(primals_582, (1824, ), (1, ))
    assert_size_stride(primals_583, (1824, ), (1, ))
    assert_size_stride(primals_584, (80, 1824), (1824, 1))
    assert_size_stride(primals_585, (80, ), (1, ))
    assert_size_stride(primals_586, (1824, 80), (80, 1))
    assert_size_stride(primals_587, (1824, ), (1, ))
    assert_size_stride(primals_588, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_589, (304, ), (1, ))
    assert_size_stride(primals_590, (304, ), (1, ))
    assert_size_stride(primals_591, (304, ), (1, ))
    assert_size_stride(primals_592, (304, ), (1, ))
    assert_size_stride(primals_593, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_594, (1824, ), (1, ))
    assert_size_stride(primals_595, (1824, ), (1, ))
    assert_size_stride(primals_596, (1824, ), (1, ))
    assert_size_stride(primals_597, (1824, ), (1, ))
    assert_size_stride(primals_598, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_599, (1824, ), (1, ))
    assert_size_stride(primals_600, (1824, ), (1, ))
    assert_size_stride(primals_601, (1824, ), (1, ))
    assert_size_stride(primals_602, (1824, ), (1, ))
    assert_size_stride(primals_603, (80, 1824), (1824, 1))
    assert_size_stride(primals_604, (80, ), (1, ))
    assert_size_stride(primals_605, (1824, 80), (80, 1))
    assert_size_stride(primals_606, (1824, ), (1, ))
    assert_size_stride(primals_607, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_608, (304, ), (1, ))
    assert_size_stride(primals_609, (304, ), (1, ))
    assert_size_stride(primals_610, (304, ), (1, ))
    assert_size_stride(primals_611, (304, ), (1, ))
    assert_size_stride(primals_612, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_613, (1824, ), (1, ))
    assert_size_stride(primals_614, (1824, ), (1, ))
    assert_size_stride(primals_615, (1824, ), (1, ))
    assert_size_stride(primals_616, (1824, ), (1, ))
    assert_size_stride(primals_617, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_618, (1824, ), (1, ))
    assert_size_stride(primals_619, (1824, ), (1, ))
    assert_size_stride(primals_620, (1824, ), (1, ))
    assert_size_stride(primals_621, (1824, ), (1, ))
    assert_size_stride(primals_622, (80, 1824), (1824, 1))
    assert_size_stride(primals_623, (80, ), (1, ))
    assert_size_stride(primals_624, (1824, 80), (80, 1))
    assert_size_stride(primals_625, (1824, ), (1, ))
    assert_size_stride(primals_626, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_627, (304, ), (1, ))
    assert_size_stride(primals_628, (304, ), (1, ))
    assert_size_stride(primals_629, (304, ), (1, ))
    assert_size_stride(primals_630, (304, ), (1, ))
    assert_size_stride(primals_631, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_632, (1824, ), (1, ))
    assert_size_stride(primals_633, (1824, ), (1, ))
    assert_size_stride(primals_634, (1824, ), (1, ))
    assert_size_stride(primals_635, (1824, ), (1, ))
    assert_size_stride(primals_636, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_637, (1824, ), (1, ))
    assert_size_stride(primals_638, (1824, ), (1, ))
    assert_size_stride(primals_639, (1824, ), (1, ))
    assert_size_stride(primals_640, (1824, ), (1, ))
    assert_size_stride(primals_641, (80, 1824), (1824, 1))
    assert_size_stride(primals_642, (80, ), (1, ))
    assert_size_stride(primals_643, (1824, 80), (80, 1))
    assert_size_stride(primals_644, (1824, ), (1, ))
    assert_size_stride(primals_645, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_646, (304, ), (1, ))
    assert_size_stride(primals_647, (304, ), (1, ))
    assert_size_stride(primals_648, (304, ), (1, ))
    assert_size_stride(primals_649, (304, ), (1, ))
    assert_size_stride(primals_650, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_651, (1824, ), (1, ))
    assert_size_stride(primals_652, (1824, ), (1, ))
    assert_size_stride(primals_653, (1824, ), (1, ))
    assert_size_stride(primals_654, (1824, ), (1, ))
    assert_size_stride(primals_655, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_656, (1824, ), (1, ))
    assert_size_stride(primals_657, (1824, ), (1, ))
    assert_size_stride(primals_658, (1824, ), (1, ))
    assert_size_stride(primals_659, (1824, ), (1, ))
    assert_size_stride(primals_660, (80, 1824), (1824, 1))
    assert_size_stride(primals_661, (80, ), (1, ))
    assert_size_stride(primals_662, (1824, 80), (80, 1))
    assert_size_stride(primals_663, (1824, ), (1, ))
    assert_size_stride(primals_664, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_665, (304, ), (1, ))
    assert_size_stride(primals_666, (304, ), (1, ))
    assert_size_stride(primals_667, (304, ), (1, ))
    assert_size_stride(primals_668, (304, ), (1, ))
    assert_size_stride(primals_669, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_670, (1824, ), (1, ))
    assert_size_stride(primals_671, (1824, ), (1, ))
    assert_size_stride(primals_672, (1824, ), (1, ))
    assert_size_stride(primals_673, (1824, ), (1, ))
    assert_size_stride(primals_674, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_675, (1824, ), (1, ))
    assert_size_stride(primals_676, (1824, ), (1, ))
    assert_size_stride(primals_677, (1824, ), (1, ))
    assert_size_stride(primals_678, (1824, ), (1, ))
    assert_size_stride(primals_679, (80, 1824), (1824, 1))
    assert_size_stride(primals_680, (80, ), (1, ))
    assert_size_stride(primals_681, (1824, 80), (80, 1))
    assert_size_stride(primals_682, (1824, ), (1, ))
    assert_size_stride(primals_683, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_684, (304, ), (1, ))
    assert_size_stride(primals_685, (304, ), (1, ))
    assert_size_stride(primals_686, (304, ), (1, ))
    assert_size_stride(primals_687, (304, ), (1, ))
    assert_size_stride(primals_688, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_689, (1824, ), (1, ))
    assert_size_stride(primals_690, (1824, ), (1, ))
    assert_size_stride(primals_691, (1824, ), (1, ))
    assert_size_stride(primals_692, (1824, ), (1, ))
    assert_size_stride(primals_693, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_694, (1824, ), (1, ))
    assert_size_stride(primals_695, (1824, ), (1, ))
    assert_size_stride(primals_696, (1824, ), (1, ))
    assert_size_stride(primals_697, (1824, ), (1, ))
    assert_size_stride(primals_698, (80, 1824), (1824, 1))
    assert_size_stride(primals_699, (80, ), (1, ))
    assert_size_stride(primals_700, (1824, 80), (80, 1))
    assert_size_stride(primals_701, (1824, ), (1, ))
    assert_size_stride(primals_702, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_703, (304, ), (1, ))
    assert_size_stride(primals_704, (304, ), (1, ))
    assert_size_stride(primals_705, (304, ), (1, ))
    assert_size_stride(primals_706, (304, ), (1, ))
    assert_size_stride(primals_707, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_708, (1824, ), (1, ))
    assert_size_stride(primals_709, (1824, ), (1, ))
    assert_size_stride(primals_710, (1824, ), (1, ))
    assert_size_stride(primals_711, (1824, ), (1, ))
    assert_size_stride(primals_712, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_713, (1824, ), (1, ))
    assert_size_stride(primals_714, (1824, ), (1, ))
    assert_size_stride(primals_715, (1824, ), (1, ))
    assert_size_stride(primals_716, (1824, ), (1, ))
    assert_size_stride(primals_717, (80, 1824), (1824, 1))
    assert_size_stride(primals_718, (80, ), (1, ))
    assert_size_stride(primals_719, (1824, 80), (80, 1))
    assert_size_stride(primals_720, (1824, ), (1, ))
    assert_size_stride(primals_721, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_722, (304, ), (1, ))
    assert_size_stride(primals_723, (304, ), (1, ))
    assert_size_stride(primals_724, (304, ), (1, ))
    assert_size_stride(primals_725, (304, ), (1, ))
    assert_size_stride(primals_726, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_727, (1824, ), (1, ))
    assert_size_stride(primals_728, (1824, ), (1, ))
    assert_size_stride(primals_729, (1824, ), (1, ))
    assert_size_stride(primals_730, (1824, ), (1, ))
    assert_size_stride(primals_731, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_732, (1824, ), (1, ))
    assert_size_stride(primals_733, (1824, ), (1, ))
    assert_size_stride(primals_734, (1824, ), (1, ))
    assert_size_stride(primals_735, (1824, ), (1, ))
    assert_size_stride(primals_736, (80, 1824), (1824, 1))
    assert_size_stride(primals_737, (80, ), (1, ))
    assert_size_stride(primals_738, (1824, 80), (80, 1))
    assert_size_stride(primals_739, (1824, ), (1, ))
    assert_size_stride(primals_740, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_741, (304, ), (1, ))
    assert_size_stride(primals_742, (304, ), (1, ))
    assert_size_stride(primals_743, (304, ), (1, ))
    assert_size_stride(primals_744, (304, ), (1, ))
    assert_size_stride(primals_745, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_746, (1824, ), (1, ))
    assert_size_stride(primals_747, (1824, ), (1, ))
    assert_size_stride(primals_748, (1824, ), (1, ))
    assert_size_stride(primals_749, (1824, ), (1, ))
    assert_size_stride(primals_750, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_751, (1824, ), (1, ))
    assert_size_stride(primals_752, (1824, ), (1, ))
    assert_size_stride(primals_753, (1824, ), (1, ))
    assert_size_stride(primals_754, (1824, ), (1, ))
    assert_size_stride(primals_755, (80, 1824), (1824, 1))
    assert_size_stride(primals_756, (80, ), (1, ))
    assert_size_stride(primals_757, (1824, 80), (80, 1))
    assert_size_stride(primals_758, (1824, ), (1, ))
    assert_size_stride(primals_759, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_760, (304, ), (1, ))
    assert_size_stride(primals_761, (304, ), (1, ))
    assert_size_stride(primals_762, (304, ), (1, ))
    assert_size_stride(primals_763, (304, ), (1, ))
    assert_size_stride(primals_764, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_765, (1824, ), (1, ))
    assert_size_stride(primals_766, (1824, ), (1, ))
    assert_size_stride(primals_767, (1824, ), (1, ))
    assert_size_stride(primals_768, (1824, ), (1, ))
    assert_size_stride(primals_769, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_770, (1824, ), (1, ))
    assert_size_stride(primals_771, (1824, ), (1, ))
    assert_size_stride(primals_772, (1824, ), (1, ))
    assert_size_stride(primals_773, (1824, ), (1, ))
    assert_size_stride(primals_774, (80, 1824), (1824, 1))
    assert_size_stride(primals_775, (80, ), (1, ))
    assert_size_stride(primals_776, (1824, 80), (80, 1))
    assert_size_stride(primals_777, (1824, ), (1, ))
    assert_size_stride(primals_778, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_779, (304, ), (1, ))
    assert_size_stride(primals_780, (304, ), (1, ))
    assert_size_stride(primals_781, (304, ), (1, ))
    assert_size_stride(primals_782, (304, ), (1, ))
    assert_size_stride(primals_783, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_784, (1824, ), (1, ))
    assert_size_stride(primals_785, (1824, ), (1, ))
    assert_size_stride(primals_786, (1824, ), (1, ))
    assert_size_stride(primals_787, (1824, ), (1, ))
    assert_size_stride(primals_788, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_789, (1824, ), (1, ))
    assert_size_stride(primals_790, (1824, ), (1, ))
    assert_size_stride(primals_791, (1824, ), (1, ))
    assert_size_stride(primals_792, (1824, ), (1, ))
    assert_size_stride(primals_793, (80, 1824), (1824, 1))
    assert_size_stride(primals_794, (80, ), (1, ))
    assert_size_stride(primals_795, (1824, 80), (80, 1))
    assert_size_stride(primals_796, (1824, ), (1, ))
    assert_size_stride(primals_797, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_798, (304, ), (1, ))
    assert_size_stride(primals_799, (304, ), (1, ))
    assert_size_stride(primals_800, (304, ), (1, ))
    assert_size_stride(primals_801, (304, ), (1, ))
    assert_size_stride(primals_802, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_803, (1824, ), (1, ))
    assert_size_stride(primals_804, (1824, ), (1, ))
    assert_size_stride(primals_805, (1824, ), (1, ))
    assert_size_stride(primals_806, (1824, ), (1, ))
    assert_size_stride(primals_807, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_808, (1824, ), (1, ))
    assert_size_stride(primals_809, (1824, ), (1, ))
    assert_size_stride(primals_810, (1824, ), (1, ))
    assert_size_stride(primals_811, (1824, ), (1, ))
    assert_size_stride(primals_812, (80, 1824), (1824, 1))
    assert_size_stride(primals_813, (80, ), (1, ))
    assert_size_stride(primals_814, (1824, 80), (80, 1))
    assert_size_stride(primals_815, (1824, ), (1, ))
    assert_size_stride(primals_816, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_817, (304, ), (1, ))
    assert_size_stride(primals_818, (304, ), (1, ))
    assert_size_stride(primals_819, (304, ), (1, ))
    assert_size_stride(primals_820, (304, ), (1, ))
    assert_size_stride(primals_821, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_822, (1824, ), (1, ))
    assert_size_stride(primals_823, (1824, ), (1, ))
    assert_size_stride(primals_824, (1824, ), (1, ))
    assert_size_stride(primals_825, (1824, ), (1, ))
    assert_size_stride(primals_826, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_827, (1824, ), (1, ))
    assert_size_stride(primals_828, (1824, ), (1, ))
    assert_size_stride(primals_829, (1824, ), (1, ))
    assert_size_stride(primals_830, (1824, ), (1, ))
    assert_size_stride(primals_831, (80, 1824), (1824, 1))
    assert_size_stride(primals_832, (80, ), (1, ))
    assert_size_stride(primals_833, (1824, 80), (80, 1))
    assert_size_stride(primals_834, (1824, ), (1, ))
    assert_size_stride(primals_835, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_836, (304, ), (1, ))
    assert_size_stride(primals_837, (304, ), (1, ))
    assert_size_stride(primals_838, (304, ), (1, ))
    assert_size_stride(primals_839, (304, ), (1, ))
    assert_size_stride(primals_840, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_841, (1824, ), (1, ))
    assert_size_stride(primals_842, (1824, ), (1, ))
    assert_size_stride(primals_843, (1824, ), (1, ))
    assert_size_stride(primals_844, (1824, ), (1, ))
    assert_size_stride(primals_845, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_846, (1824, ), (1, ))
    assert_size_stride(primals_847, (1824, ), (1, ))
    assert_size_stride(primals_848, (1824, ), (1, ))
    assert_size_stride(primals_849, (1824, ), (1, ))
    assert_size_stride(primals_850, (80, 1824), (1824, 1))
    assert_size_stride(primals_851, (80, ), (1, ))
    assert_size_stride(primals_852, (1824, 80), (80, 1))
    assert_size_stride(primals_853, (1824, ), (1, ))
    assert_size_stride(primals_854, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_855, (304, ), (1, ))
    assert_size_stride(primals_856, (304, ), (1, ))
    assert_size_stride(primals_857, (304, ), (1, ))
    assert_size_stride(primals_858, (304, ), (1, ))
    assert_size_stride(primals_859, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_860, (1824, ), (1, ))
    assert_size_stride(primals_861, (1824, ), (1, ))
    assert_size_stride(primals_862, (1824, ), (1, ))
    assert_size_stride(primals_863, (1824, ), (1, ))
    assert_size_stride(primals_864, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_865, (1824, ), (1, ))
    assert_size_stride(primals_866, (1824, ), (1, ))
    assert_size_stride(primals_867, (1824, ), (1, ))
    assert_size_stride(primals_868, (1824, ), (1, ))
    assert_size_stride(primals_869, (80, 1824), (1824, 1))
    assert_size_stride(primals_870, (80, ), (1, ))
    assert_size_stride(primals_871, (1824, 80), (80, 1))
    assert_size_stride(primals_872, (1824, ), (1, ))
    assert_size_stride(primals_873, (304, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_874, (304, ), (1, ))
    assert_size_stride(primals_875, (304, ), (1, ))
    assert_size_stride(primals_876, (304, ), (1, ))
    assert_size_stride(primals_877, (304, ), (1, ))
    assert_size_stride(primals_878, (1824, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(primals_879, (1824, ), (1, ))
    assert_size_stride(primals_880, (1824, ), (1, ))
    assert_size_stride(primals_881, (1824, ), (1, ))
    assert_size_stride(primals_882, (1824, ), (1, ))
    assert_size_stride(primals_883, (1824, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_884, (1824, ), (1, ))
    assert_size_stride(primals_885, (1824, ), (1, ))
    assert_size_stride(primals_886, (1824, ), (1, ))
    assert_size_stride(primals_887, (1824, ), (1, ))
    assert_size_stride(primals_888, (80, 1824), (1824, 1))
    assert_size_stride(primals_889, (80, ), (1, ))
    assert_size_stride(primals_890, (1824, 80), (80, 1))
    assert_size_stride(primals_891, (1824, ), (1, ))
    assert_size_stride(primals_892, (512, 1824, 1, 1), (1824, 1, 1, 1))
    assert_size_stride(primals_893, (512, ), (1, ))
    assert_size_stride(primals_894, (512, ), (1, ))
    assert_size_stride(primals_895, (512, ), (1, ))
    assert_size_stride(primals_896, (512, ), (1, ))
    assert_size_stride(primals_897, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_898, (3072, ), (1, ))
    assert_size_stride(primals_899, (3072, ), (1, ))
    assert_size_stride(primals_900, (3072, ), (1, ))
    assert_size_stride(primals_901, (3072, ), (1, ))
    assert_size_stride(primals_902, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_903, (3072, ), (1, ))
    assert_size_stride(primals_904, (3072, ), (1, ))
    assert_size_stride(primals_905, (3072, ), (1, ))
    assert_size_stride(primals_906, (3072, ), (1, ))
    assert_size_stride(primals_907, (128, 3072), (3072, 1))
    assert_size_stride(primals_908, (128, ), (1, ))
    assert_size_stride(primals_909, (3072, 128), (128, 1))
    assert_size_stride(primals_910, (3072, ), (1, ))
    assert_size_stride(primals_911, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_912, (512, ), (1, ))
    assert_size_stride(primals_913, (512, ), (1, ))
    assert_size_stride(primals_914, (512, ), (1, ))
    assert_size_stride(primals_915, (512, ), (1, ))
    assert_size_stride(primals_916, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_917, (3072, ), (1, ))
    assert_size_stride(primals_918, (3072, ), (1, ))
    assert_size_stride(primals_919, (3072, ), (1, ))
    assert_size_stride(primals_920, (3072, ), (1, ))
    assert_size_stride(primals_921, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_922, (3072, ), (1, ))
    assert_size_stride(primals_923, (3072, ), (1, ))
    assert_size_stride(primals_924, (3072, ), (1, ))
    assert_size_stride(primals_925, (3072, ), (1, ))
    assert_size_stride(primals_926, (128, 3072), (3072, 1))
    assert_size_stride(primals_927, (128, ), (1, ))
    assert_size_stride(primals_928, (3072, 128), (128, 1))
    assert_size_stride(primals_929, (3072, ), (1, ))
    assert_size_stride(primals_930, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_931, (512, ), (1, ))
    assert_size_stride(primals_932, (512, ), (1, ))
    assert_size_stride(primals_933, (512, ), (1, ))
    assert_size_stride(primals_934, (512, ), (1, ))
    assert_size_stride(primals_935, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_936, (3072, ), (1, ))
    assert_size_stride(primals_937, (3072, ), (1, ))
    assert_size_stride(primals_938, (3072, ), (1, ))
    assert_size_stride(primals_939, (3072, ), (1, ))
    assert_size_stride(primals_940, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_941, (3072, ), (1, ))
    assert_size_stride(primals_942, (3072, ), (1, ))
    assert_size_stride(primals_943, (3072, ), (1, ))
    assert_size_stride(primals_944, (3072, ), (1, ))
    assert_size_stride(primals_945, (128, 3072), (3072, 1))
    assert_size_stride(primals_946, (128, ), (1, ))
    assert_size_stride(primals_947, (3072, 128), (128, 1))
    assert_size_stride(primals_948, (3072, ), (1, ))
    assert_size_stride(primals_949, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_950, (512, ), (1, ))
    assert_size_stride(primals_951, (512, ), (1, ))
    assert_size_stride(primals_952, (512, ), (1, ))
    assert_size_stride(primals_953, (512, ), (1, ))
    assert_size_stride(primals_954, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_955, (3072, ), (1, ))
    assert_size_stride(primals_956, (3072, ), (1, ))
    assert_size_stride(primals_957, (3072, ), (1, ))
    assert_size_stride(primals_958, (3072, ), (1, ))
    assert_size_stride(primals_959, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_960, (3072, ), (1, ))
    assert_size_stride(primals_961, (3072, ), (1, ))
    assert_size_stride(primals_962, (3072, ), (1, ))
    assert_size_stride(primals_963, (3072, ), (1, ))
    assert_size_stride(primals_964, (128, 3072), (3072, 1))
    assert_size_stride(primals_965, (128, ), (1, ))
    assert_size_stride(primals_966, (3072, 128), (128, 1))
    assert_size_stride(primals_967, (3072, ), (1, ))
    assert_size_stride(primals_968, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_969, (512, ), (1, ))
    assert_size_stride(primals_970, (512, ), (1, ))
    assert_size_stride(primals_971, (512, ), (1, ))
    assert_size_stride(primals_972, (512, ), (1, ))
    assert_size_stride(primals_973, (1792, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_974, (1792, ), (1, ))
    assert_size_stride(primals_975, (1792, ), (1, ))
    assert_size_stride(primals_976, (1792, ), (1, ))
    assert_size_stride(primals_977, (1792, ), (1, ))
    assert_size_stride(primals_978, (1000, 1792), (1792, 1))
    assert_size_stride(primals_979, (1000, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((24, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 72, 9, grid=grid(72, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((24, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_7, buf2, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_7
        buf3 = empty_strided_cuda((24, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_17, buf3, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_17
        buf4 = empty_strided_cuda((24, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_27, buf4, 576, 9, grid=grid(576, 9), stream=stream0)
        del primals_27
        buf5 = empty_strided_cuda((96, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_37, buf5, 2304, 9, grid=grid(2304, 9), stream=stream0)
        del primals_37
        buf6 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_47, buf6, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_47
        buf7 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_57, buf7, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_57
        buf8 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_67, buf8, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_67
        buf9 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_77, buf9, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_77
        buf10 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_87, buf10, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_87
        buf11 = empty_strided_cuda((320, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_97, buf11, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_97
        buf12 = empty_strided_cuda((320, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_107, buf12, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_107
        buf13 = empty_strided_cuda((320, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_117, buf13, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_117
        buf14 = empty_strided_cuda((320, 80, 3, 3), (720, 1, 240, 80), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_127, buf14, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_127
        # Topologically Sorted Source Nodes: [features_0_0], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf16 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [features_0_1, sigmoid_1, mul_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf17, buf15, primals_3, primals_4, primals_5, primals_6, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_0], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf19 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [features_1_conv_1, sigmoid_2, mul_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf20, buf18, primals_8, primals_9, primals_10, primals_11, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_3], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf22 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        # Topologically Sorted Source Nodes: [features_1_conv_4, add_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf17, buf21, primals_13, primals_14, primals_15, primals_16, buf22, 98304, grid=grid(98304), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [features_2_conv_0], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf24 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [features_2_conv_1, sigmoid_3, mul_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf25, buf23, primals_18, primals_19, primals_20, primals_21, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_2_conv_3], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf27 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        # Topologically Sorted Source Nodes: [features_2_conv_4, add_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf22, buf26, primals_23, primals_24, primals_25, primals_26, buf27, 98304, grid=grid(98304), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [features_3_conv_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf29 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [features_3_conv_1, sigmoid_4, mul_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf30, buf28, primals_28, primals_29, primals_30, primals_31, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_3_conv_3], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf32 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        # Topologically Sorted Source Nodes: [features_3_conv_4, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf27, buf31, primals_33, primals_34, primals_35, primals_36, buf32, 98304, grid=grid(98304), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [features_4_conv_0], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, buf5, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 96, 16, 16), (24576, 1, 1536, 96))
        buf34 = empty_strided_cuda((4, 96, 16, 16), (24576, 1, 1536, 96), torch.float32)
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [features_4_conv_1, sigmoid_5, mul_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_8.run(buf35, buf33, primals_38, primals_39, primals_40, primals_41, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_4_conv_3], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf37 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_4_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf36, primals_43, primals_44, primals_45, primals_46, buf37, 49152, grid=grid(49152), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [features_5_conv_0], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 192, 16, 16), (49152, 1, 3072, 192))
        buf39 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [features_5_conv_1, sigmoid_6, mul_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf40, buf38, primals_48, primals_49, primals_50, primals_51, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [features_5_conv_3], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf42 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_5_conv_4, add_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf37, buf41, primals_53, primals_54, primals_55, primals_56, buf42, 49152, grid=grid(49152), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [features_6_conv_0], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 192, 16, 16), (49152, 1, 3072, 192))
        buf44 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [features_6_conv_1, sigmoid_7, mul_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf45, buf43, primals_58, primals_59, primals_60, primals_61, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [features_6_conv_3], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf47 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_6_conv_4, add_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf42, buf46, primals_63, primals_64, primals_65, primals_66, buf47, 49152, grid=grid(49152), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [features_7_conv_0], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 192, 16, 16), (49152, 1, 3072, 192))
        buf49 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [features_7_conv_1, sigmoid_8, mul_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf50, buf48, primals_68, primals_69, primals_70, primals_71, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [features_7_conv_3], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf52 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_7_conv_4, add_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf47, buf51, primals_73, primals_74, primals_75, primals_76, buf52, 49152, grid=grid(49152), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [features_8_conv_0], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 192, 16, 16), (49152, 1, 3072, 192))
        buf54 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [features_8_conv_1, sigmoid_9, mul_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf55, buf53, primals_78, primals_79, primals_80, primals_81, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [features_8_conv_3], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf57 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_8_conv_4, add_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf52, buf56, primals_83, primals_84, primals_85, primals_86, buf57, 49152, grid=grid(49152), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [features_9_conv_0], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, buf10, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 192, 8, 8), (12288, 1, 1536, 192))
        buf59 = empty_strided_cuda((4, 192, 8, 8), (12288, 1, 1536, 192), torch.float32)
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [features_9_conv_1, sigmoid_10, mul_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_12.run(buf60, buf58, primals_88, primals_89, primals_90, primals_91, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_9_conv_3], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 80, 8, 8), (5120, 1, 640, 80))
        buf62 = empty_strided_cuda((4, 80, 8, 8), (5120, 1, 640, 80), torch.float32)
        # Topologically Sorted Source Nodes: [features_9_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf61, primals_93, primals_94, primals_95, primals_96, buf62, 20480, grid=grid(20480), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [features_10_conv_0], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 320, 8, 8), (20480, 1, 2560, 320))
        buf64 = empty_strided_cuda((4, 320, 8, 8), (20480, 1, 2560, 320), torch.float32)
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [features_10_conv_1, sigmoid_11, mul_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf65, buf63, primals_98, primals_99, primals_100, primals_101, 81920, grid=grid(81920), stream=stream0)
        # Topologically Sorted Source Nodes: [features_10_conv_3], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 80, 8, 8), (5120, 1, 640, 80))
        buf67 = empty_strided_cuda((4, 80, 8, 8), (5120, 1, 640, 80), torch.float32)
        # Topologically Sorted Source Nodes: [features_10_conv_4, add_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf62, buf66, primals_103, primals_104, primals_105, primals_106, buf67, 20480, grid=grid(20480), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [features_11_conv_0], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 320, 8, 8), (20480, 1, 2560, 320))
        buf69 = empty_strided_cuda((4, 320, 8, 8), (20480, 1, 2560, 320), torch.float32)
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [features_11_conv_1, sigmoid_12, mul_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf70, buf68, primals_108, primals_109, primals_110, primals_111, 81920, grid=grid(81920), stream=stream0)
        # Topologically Sorted Source Nodes: [features_11_conv_3], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 80, 8, 8), (5120, 1, 640, 80))
        buf72 = empty_strided_cuda((4, 80, 8, 8), (5120, 1, 640, 80), torch.float32)
        # Topologically Sorted Source Nodes: [features_11_conv_4, add_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf67, buf71, primals_113, primals_114, primals_115, primals_116, buf72, 20480, grid=grid(20480), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [features_12_conv_0], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 320, 8, 8), (20480, 1, 2560, 320))
        buf74 = empty_strided_cuda((4, 320, 8, 8), (20480, 1, 2560, 320), torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [features_12_conv_1, sigmoid_13, mul_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf75, buf73, primals_118, primals_119, primals_120, primals_121, 81920, grid=grid(81920), stream=stream0)
        # Topologically Sorted Source Nodes: [features_12_conv_3], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 80, 8, 8), (5120, 1, 640, 80))
        buf77 = empty_strided_cuda((4, 80, 8, 8), (5120, 1, 640, 80), torch.float32)
        # Topologically Sorted Source Nodes: [features_12_conv_4, add_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf72, buf76, primals_123, primals_124, primals_125, primals_126, buf77, 20480, grid=grid(20480), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [features_13_conv_0], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 320, 8, 8), (20480, 1, 2560, 320))
        buf79 = empty_strided_cuda((4, 320, 8, 8), (20480, 1, 2560, 320), torch.float32)
        buf80 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [features_13_conv_1, sigmoid_14, mul_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf80, buf78, primals_128, primals_129, primals_130, primals_131, 81920, grid=grid(81920), stream=stream0)
        # Topologically Sorted Source Nodes: [features_13_conv_3], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 80, 8, 8), (5120, 1, 640, 80))
        buf82 = empty_strided_cuda((4, 80, 8, 8), (5120, 1, 640, 80), torch.float32)
        # Topologically Sorted Source Nodes: [features_13_conv_4, add_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf77, buf81, primals_133, primals_134, primals_135, primals_136, buf82, 20480, grid=grid(20480), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [features_14_conv_0], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 320, 8, 8), (20480, 1, 2560, 320))
        buf84 = empty_strided_cuda((4, 320, 8, 8), (20480, 1, 2560, 320), torch.float32)
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [features_14_conv_1, sigmoid_15, mul_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf85, buf83, primals_138, primals_139, primals_140, primals_141, 81920, grid=grid(81920), stream=stream0)
        # Topologically Sorted Source Nodes: [features_14_conv_3], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_142, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=320, bias=None)
        assert_size_stride(buf86, (4, 320, 4, 4), (5120, 1, 1280, 320))
        buf87 = empty_strided_cuda((4, 320, 4, 4), (5120, 1, 1280, 320), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf86, primals_143, primals_144, primals_145, primals_146, buf87, 20480, grid=grid(20480), stream=stream0)
        buf88 = empty_strided_cuda((4, 320, 1, 1), (320, 1, 1280, 1280), torch.float32)
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_16, mul_16, features_14_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_17.run(buf89, buf87, 1280, 16, grid=grid(1280), stream=stream0)
        buf90 = empty_strided_cuda((4, 24), (24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_148, reinterpret_tensor(buf89, (4, 320), (320, 1), 0), reinterpret_tensor(primals_147, (320, 24), (1, 320), 0), alpha=1, beta=1, out=buf90)
        del primals_148
        buf91 = empty_strided_cuda((4, 24), (24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_17, mul_17], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_18.run(buf90, buf91, 96, grid=grid(96), stream=stream0)
        buf92 = empty_strided_cuda((4, 320), (320, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_150, buf91, reinterpret_tensor(primals_149, (24, 320), (1, 24), 0), alpha=1, beta=1, out=buf92)
        del primals_150
        buf93 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_16, mul_16, mul_18], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_19.run(buf93, buf92, 20480, grid=grid(20480), stream=stream0)
        # Topologically Sorted Source Nodes: [features_14_conv_7], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf95 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf94, primals_152, primals_153, primals_154, primals_155, buf95, 10240, grid=grid(10240), stream=stream0)
        del primals_155
        # Topologically Sorted Source Nodes: [features_15_conv_0], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf97 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        buf98 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [features_15_conv_1, sigmoid_20, mul_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf98, buf96, primals_157, primals_158, primals_159, primals_160, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_15_conv_3], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=640, bias=None)
        assert_size_stride(buf99, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf100 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf99, primals_162, primals_163, primals_164, primals_165, buf100, 40960, grid=grid(40960), stream=stream0)
        buf101 = empty_strided_cuda((4, 640, 1, 1), (640, 1, 2560, 2560), torch.float32)
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_22, mul_20, features_15_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf102, buf100, 2560, 16, grid=grid(2560), stream=stream0)
        buf103 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_167, reinterpret_tensor(buf102, (4, 640), (640, 1), 0), reinterpret_tensor(primals_166, (640, 40), (1, 640), 0), alpha=1, beta=1, out=buf103)
        del primals_167
        buf104 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_24, mul_21], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf103, buf104, 160, grid=grid(160), stream=stream0)
        buf105 = empty_strided_cuda((4, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_169, buf104, reinterpret_tensor(primals_168, (40, 640), (1, 40), 0), alpha=1, beta=1, out=buf105)
        del primals_169
        buf106 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_22, mul_20, mul_22], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf106, buf105, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_15_conv_7], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf108 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_8, add_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf95, buf107, primals_171, primals_172, primals_173, primals_174, buf108, 10240, grid=grid(10240), stream=stream0)
        del primals_174
        # Topologically Sorted Source Nodes: [features_16_conv_0], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf110 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        buf111 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [features_16_conv_1, sigmoid_27, mul_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf111, buf109, primals_176, primals_177, primals_178, primals_179, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_16_conv_3], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=640, bias=None)
        assert_size_stride(buf112, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf113 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf112, primals_181, primals_182, primals_183, primals_184, buf113, 40960, grid=grid(40960), stream=stream0)
        buf114 = empty_strided_cuda((4, 640, 1, 1), (640, 1, 2560, 2560), torch.float32)
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_29, mul_24, features_16_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf115, buf113, 2560, 16, grid=grid(2560), stream=stream0)
        buf116 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_186, reinterpret_tensor(buf115, (4, 640), (640, 1), 0), reinterpret_tensor(primals_185, (640, 40), (1, 640), 0), alpha=1, beta=1, out=buf116)
        del primals_186
        buf117 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_31, mul_25], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf116, buf117, 160, grid=grid(160), stream=stream0)
        buf118 = empty_strided_cuda((4, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_188, buf117, reinterpret_tensor(primals_187, (40, 640), (1, 40), 0), alpha=1, beta=1, out=buf118)
        del primals_188
        buf119 = buf113; del buf113  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_29, mul_24, mul_26], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf119, buf118, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_16_conv_7], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf121 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_8, add_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf108, buf120, primals_190, primals_191, primals_192, primals_193, buf121, 10240, grid=grid(10240), stream=stream0)
        del primals_193
        # Topologically Sorted Source Nodes: [features_17_conv_0], Original ATen: [aten.convolution]
        buf122 = extern_kernels.convolution(buf121, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf123 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        buf124 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [features_17_conv_1, sigmoid_34, mul_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf124, buf122, primals_195, primals_196, primals_197, primals_198, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_17_conv_3], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=640, bias=None)
        assert_size_stride(buf125, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf126 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf125, primals_200, primals_201, primals_202, primals_203, buf126, 40960, grid=grid(40960), stream=stream0)
        buf127 = empty_strided_cuda((4, 640, 1, 1), (640, 1, 2560, 2560), torch.float32)
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_36, mul_28, features_17_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf128, buf126, 2560, 16, grid=grid(2560), stream=stream0)
        buf129 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_205, reinterpret_tensor(buf128, (4, 640), (640, 1), 0), reinterpret_tensor(primals_204, (640, 40), (1, 640), 0), alpha=1, beta=1, out=buf129)
        del primals_205
        buf130 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_38, mul_29], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf129, buf130, 160, grid=grid(160), stream=stream0)
        buf131 = empty_strided_cuda((4, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_207, buf130, reinterpret_tensor(primals_206, (40, 640), (1, 40), 0), alpha=1, beta=1, out=buf131)
        del primals_207
        buf132 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_36, mul_28, mul_30], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf132, buf131, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_17_conv_7], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf134 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_8, add_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf121, buf133, primals_209, primals_210, primals_211, primals_212, buf134, 10240, grid=grid(10240), stream=stream0)
        del primals_212
        # Topologically Sorted Source Nodes: [features_18_conv_0], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf136 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        buf137 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [features_18_conv_1, sigmoid_41, mul_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf137, buf135, primals_214, primals_215, primals_216, primals_217, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_18_conv_3], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=640, bias=None)
        assert_size_stride(buf138, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf139 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf138, primals_219, primals_220, primals_221, primals_222, buf139, 40960, grid=grid(40960), stream=stream0)
        buf140 = empty_strided_cuda((4, 640, 1, 1), (640, 1, 2560, 2560), torch.float32)
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_43, mul_32, features_18_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf141, buf139, 2560, 16, grid=grid(2560), stream=stream0)
        buf142 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_224, reinterpret_tensor(buf141, (4, 640), (640, 1), 0), reinterpret_tensor(primals_223, (640, 40), (1, 640), 0), alpha=1, beta=1, out=buf142)
        del primals_224
        buf143 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_45, mul_33], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf142, buf143, 160, grid=grid(160), stream=stream0)
        buf144 = empty_strided_cuda((4, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_226, buf143, reinterpret_tensor(primals_225, (40, 640), (1, 40), 0), alpha=1, beta=1, out=buf144)
        del primals_226
        buf145 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_43, mul_32, mul_34], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf145, buf144, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_18_conv_7], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf147 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_8, add_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf134, buf146, primals_228, primals_229, primals_230, primals_231, buf147, 10240, grid=grid(10240), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [features_19_conv_0], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf149 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        buf150 = buf149; del buf149  # reuse
        # Topologically Sorted Source Nodes: [features_19_conv_1, sigmoid_48, mul_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf150, buf148, primals_233, primals_234, primals_235, primals_236, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_19_conv_3], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=640, bias=None)
        assert_size_stride(buf151, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf152 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf151, primals_238, primals_239, primals_240, primals_241, buf152, 40960, grid=grid(40960), stream=stream0)
        buf153 = empty_strided_cuda((4, 640, 1, 1), (640, 1, 2560, 2560), torch.float32)
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_50, mul_36, features_19_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf154, buf152, 2560, 16, grid=grid(2560), stream=stream0)
        buf155 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_243, reinterpret_tensor(buf154, (4, 640), (640, 1), 0), reinterpret_tensor(primals_242, (640, 40), (1, 640), 0), alpha=1, beta=1, out=buf155)
        del primals_243
        buf156 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_52, mul_37], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf155, buf156, 160, grid=grid(160), stream=stream0)
        buf157 = empty_strided_cuda((4, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_245, buf156, reinterpret_tensor(primals_244, (40, 640), (1, 40), 0), alpha=1, beta=1, out=buf157)
        del primals_245
        buf158 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_50, mul_36, mul_38], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf158, buf157, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_19_conv_7], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf160 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_8, add_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf147, buf159, primals_247, primals_248, primals_249, primals_250, buf160, 10240, grid=grid(10240), stream=stream0)
        del primals_250
        # Topologically Sorted Source Nodes: [features_20_conv_0], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf162 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        buf163 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [features_20_conv_1, sigmoid_55, mul_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf163, buf161, primals_252, primals_253, primals_254, primals_255, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_20_conv_3], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=640, bias=None)
        assert_size_stride(buf164, (4, 640, 4, 4), (10240, 1, 2560, 640))
        buf165 = empty_strided_cuda((4, 640, 4, 4), (10240, 1, 2560, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf164, primals_257, primals_258, primals_259, primals_260, buf165, 40960, grid=grid(40960), stream=stream0)
        buf166 = empty_strided_cuda((4, 640, 1, 1), (640, 1, 2560, 2560), torch.float32)
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_57, mul_40, features_20_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf167, buf165, 2560, 16, grid=grid(2560), stream=stream0)
        buf168 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_262, reinterpret_tensor(buf167, (4, 640), (640, 1), 0), reinterpret_tensor(primals_261, (640, 40), (1, 640), 0), alpha=1, beta=1, out=buf168)
        del primals_262
        buf169 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_59, mul_41], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf168, buf169, 160, grid=grid(160), stream=stream0)
        buf170 = empty_strided_cuda((4, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_264, buf169, reinterpret_tensor(primals_263, (40, 640), (1, 40), 0), alpha=1, beta=1, out=buf170)
        del primals_264
        buf171 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_57, mul_40, mul_42], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf171, buf170, 40960, grid=grid(40960), stream=stream0)
        # Topologically Sorted Source Nodes: [features_20_conv_7], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf173 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_8, add_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf160, buf172, primals_266, primals_267, primals_268, primals_269, buf173, 10240, grid=grid(10240), stream=stream0)
        del primals_269
        # Topologically Sorted Source Nodes: [features_21_conv_0], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf175 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf176 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [features_21_conv_1, sigmoid_62, mul_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_27.run(buf176, buf174, primals_271, primals_272, primals_273, primals_274, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_3], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf177, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf178 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf177, primals_276, primals_277, primals_278, primals_279, buf178, 61440, grid=grid(61440), stream=stream0)
        buf179 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_64, mul_44, features_21_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_29.run(buf180, buf178, 3840, 16, grid=grid(3840), stream=stream0)
        buf181 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_281, reinterpret_tensor(buf180, (4, 960), (960, 1), 0), reinterpret_tensor(primals_280, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf181)
        del primals_281
        buf182 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_66, mul_45], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf181, buf182, 160, grid=grid(160), stream=stream0)
        buf183 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_283, buf182, reinterpret_tensor(primals_282, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf183)
        del primals_283
        buf184 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_64, mul_44, mul_46], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf184, buf183, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_7], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_284, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf186 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf185, primals_285, primals_286, primals_287, primals_288, buf186, 11264, grid=grid(11264), stream=stream0)
        del primals_288
        # Topologically Sorted Source Nodes: [features_22_conv_0], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf188 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf189 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [features_22_conv_1, sigmoid_69, mul_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf189, buf187, primals_290, primals_291, primals_292, primals_293, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_3], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf190, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf191 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf190, primals_295, primals_296, primals_297, primals_298, buf191, 67584, grid=grid(67584), stream=stream0)
        buf192 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf193 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_71, mul_48, features_22_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf193, buf191, 4224, 16, grid=grid(4224), stream=stream0)
        buf194 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_300, reinterpret_tensor(buf193, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_299, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf194)
        del primals_300
        buf195 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_73, mul_49], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf194, buf195, 192, grid=grid(192), stream=stream0)
        buf196 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_302, buf195, reinterpret_tensor(primals_301, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf196)
        del primals_302
        buf197 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_71, mul_48, mul_50], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf197, buf196, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_7], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf199 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_8, add_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf186, buf198, primals_304, primals_305, primals_306, primals_307, buf199, 11264, grid=grid(11264), stream=stream0)
        del primals_307
        # Topologically Sorted Source Nodes: [features_23_conv_0], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf201 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf202 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [features_23_conv_1, sigmoid_76, mul_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf202, buf200, primals_309, primals_310, primals_311, primals_312, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_3], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_313, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf203, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf204 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf203, primals_314, primals_315, primals_316, primals_317, buf204, 67584, grid=grid(67584), stream=stream0)
        buf205 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf206 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_78, mul_52, features_23_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf206, buf204, 4224, 16, grid=grid(4224), stream=stream0)
        buf207 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_319, reinterpret_tensor(buf206, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_318, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf207)
        del primals_319
        buf208 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_80, mul_53], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf207, buf208, 192, grid=grid(192), stream=stream0)
        buf209 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_321, buf208, reinterpret_tensor(primals_320, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf209)
        del primals_321
        buf210 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_78, mul_52, mul_54], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf210, buf209, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_7], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf212 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_8, add_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf199, buf211, primals_323, primals_324, primals_325, primals_326, buf212, 11264, grid=grid(11264), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [features_24_conv_0], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf214 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf215 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [features_24_conv_1, sigmoid_83, mul_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf215, buf213, primals_328, primals_329, primals_330, primals_331, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_3], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf216, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf217 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf216, primals_333, primals_334, primals_335, primals_336, buf217, 67584, grid=grid(67584), stream=stream0)
        buf218 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_85, mul_56, features_24_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf219, buf217, 4224, 16, grid=grid(4224), stream=stream0)
        buf220 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_338, reinterpret_tensor(buf219, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_337, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf220)
        del primals_338
        buf221 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_87, mul_57], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf220, buf221, 192, grid=grid(192), stream=stream0)
        buf222 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_340, buf221, reinterpret_tensor(primals_339, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf222)
        del primals_340
        buf223 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_85, mul_56, mul_58], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf223, buf222, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_7], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf225 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_8, add_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf212, buf224, primals_342, primals_343, primals_344, primals_345, buf225, 11264, grid=grid(11264), stream=stream0)
        del primals_345
        # Topologically Sorted Source Nodes: [features_25_conv_0], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_346, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf227 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf228 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [features_25_conv_1, sigmoid_90, mul_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf228, buf226, primals_347, primals_348, primals_349, primals_350, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_3], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_351, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf229, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf230 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf229, primals_352, primals_353, primals_354, primals_355, buf230, 67584, grid=grid(67584), stream=stream0)
        buf231 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf232 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_92, mul_60, features_25_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf232, buf230, 4224, 16, grid=grid(4224), stream=stream0)
        buf233 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_357, reinterpret_tensor(buf232, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_356, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf233)
        del primals_357
        buf234 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_94, mul_61], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf233, buf234, 192, grid=grid(192), stream=stream0)
        buf235 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_359, buf234, reinterpret_tensor(primals_358, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf235)
        del primals_359
        buf236 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_92, mul_60, mul_62], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf236, buf235, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_7], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_360, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf238 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_8, add_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf225, buf237, primals_361, primals_362, primals_363, primals_364, buf238, 11264, grid=grid(11264), stream=stream0)
        del primals_364
        # Topologically Sorted Source Nodes: [features_26_conv_0], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_365, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf240 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf241 = buf240; del buf240  # reuse
        # Topologically Sorted Source Nodes: [features_26_conv_1, sigmoid_97, mul_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf241, buf239, primals_366, primals_367, primals_368, primals_369, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_3], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_370, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf242, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf243 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf242, primals_371, primals_372, primals_373, primals_374, buf243, 67584, grid=grid(67584), stream=stream0)
        buf244 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_99, mul_64, features_26_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf245, buf243, 4224, 16, grid=grid(4224), stream=stream0)
        buf246 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_376, reinterpret_tensor(buf245, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_375, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf246)
        del primals_376
        buf247 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_101, mul_65], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf246, buf247, 192, grid=grid(192), stream=stream0)
        buf248 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_378, buf247, reinterpret_tensor(primals_377, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf248)
        del primals_378
        buf249 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_99, mul_64, mul_66], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf249, buf248, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_7], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_379, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf251 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_8, add_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf238, buf250, primals_380, primals_381, primals_382, primals_383, buf251, 11264, grid=grid(11264), stream=stream0)
        del primals_383
        # Topologically Sorted Source Nodes: [features_27_conv_0], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf251, primals_384, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf253 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf254 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [features_27_conv_1, sigmoid_104, mul_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf254, buf252, primals_385, primals_386, primals_387, primals_388, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_3], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_389, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf255, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf256 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf255, primals_390, primals_391, primals_392, primals_393, buf256, 67584, grid=grid(67584), stream=stream0)
        buf257 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_106, mul_68, features_27_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf258, buf256, 4224, 16, grid=grid(4224), stream=stream0)
        buf259 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_395, reinterpret_tensor(buf258, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_394, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf259)
        del primals_395
        buf260 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_108, mul_69], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf259, buf260, 192, grid=grid(192), stream=stream0)
        buf261 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_397, buf260, reinterpret_tensor(primals_396, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf261)
        del primals_397
        buf262 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_106, mul_68, mul_70], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf262, buf261, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_7], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_398, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf264 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_8, add_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf251, buf263, primals_399, primals_400, primals_401, primals_402, buf264, 11264, grid=grid(11264), stream=stream0)
        del primals_402
        # Topologically Sorted Source Nodes: [features_28_conv_0], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_403, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf266 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf267 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [features_28_conv_1, sigmoid_111, mul_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf267, buf265, primals_404, primals_405, primals_406, primals_407, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_3], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_408, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf268, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf269 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf268, primals_409, primals_410, primals_411, primals_412, buf269, 67584, grid=grid(67584), stream=stream0)
        buf270 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_113, mul_72, features_28_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf271, buf269, 4224, 16, grid=grid(4224), stream=stream0)
        buf272 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_414, reinterpret_tensor(buf271, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_413, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf272)
        del primals_414
        buf273 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_115, mul_73], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf272, buf273, 192, grid=grid(192), stream=stream0)
        buf274 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_416, buf273, reinterpret_tensor(primals_415, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf274)
        del primals_416
        buf275 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_113, mul_72, mul_74], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf275, buf274, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_7], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_417, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf277 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_8, add_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf264, buf276, primals_418, primals_419, primals_420, primals_421, buf277, 11264, grid=grid(11264), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [features_29_conv_0], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf279 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf280 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [features_29_conv_1, sigmoid_118, mul_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf280, buf278, primals_423, primals_424, primals_425, primals_426, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_3], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf281, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf282 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf281, primals_428, primals_429, primals_430, primals_431, buf282, 67584, grid=grid(67584), stream=stream0)
        buf283 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_120, mul_76, features_29_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf284, buf282, 4224, 16, grid=grid(4224), stream=stream0)
        buf285 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_433, reinterpret_tensor(buf284, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_432, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf285)
        del primals_433
        buf286 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_122, mul_77], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf285, buf286, 192, grid=grid(192), stream=stream0)
        buf287 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_435, buf286, reinterpret_tensor(primals_434, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf287)
        del primals_435
        buf288 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_120, mul_76, mul_78], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf288, buf287, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_7], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, primals_436, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf290 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_8, add_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf277, buf289, primals_437, primals_438, primals_439, primals_440, buf290, 11264, grid=grid(11264), stream=stream0)
        del primals_440
        # Topologically Sorted Source Nodes: [features_30_conv_0], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_441, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf292 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf293 = buf292; del buf292  # reuse
        # Topologically Sorted Source Nodes: [features_30_conv_1, sigmoid_125, mul_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf293, buf291, primals_442, primals_443, primals_444, primals_445, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_3], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_446, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf294, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf295 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf294, primals_447, primals_448, primals_449, primals_450, buf295, 67584, grid=grid(67584), stream=stream0)
        buf296 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_127, mul_80, features_30_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf297, buf295, 4224, 16, grid=grid(4224), stream=stream0)
        buf298 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_452, reinterpret_tensor(buf297, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_451, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf298)
        del primals_452
        buf299 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_129, mul_81], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf298, buf299, 192, grid=grid(192), stream=stream0)
        buf300 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_454, buf299, reinterpret_tensor(primals_453, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf300)
        del primals_454
        buf301 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_127, mul_80, mul_82], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf301, buf300, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_7], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_455, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf303 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_8, add_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf290, buf302, primals_456, primals_457, primals_458, primals_459, buf303, 11264, grid=grid(11264), stream=stream0)
        del primals_459
        # Topologically Sorted Source Nodes: [features_31_conv_0], Original ATen: [aten.convolution]
        buf304 = extern_kernels.convolution(buf303, primals_460, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf304, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf305 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf306 = buf305; del buf305  # reuse
        # Topologically Sorted Source Nodes: [features_31_conv_1, sigmoid_132, mul_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf306, buf304, primals_461, primals_462, primals_463, primals_464, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_3], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_465, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf307, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf308 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf307, primals_466, primals_467, primals_468, primals_469, buf308, 67584, grid=grid(67584), stream=stream0)
        buf309 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf310 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_134, mul_84, features_31_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf310, buf308, 4224, 16, grid=grid(4224), stream=stream0)
        buf311 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_471, reinterpret_tensor(buf310, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_470, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf311)
        del primals_471
        buf312 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_136, mul_85], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf311, buf312, 192, grid=grid(192), stream=stream0)
        buf313 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_473, buf312, reinterpret_tensor(primals_472, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf313)
        del primals_473
        buf314 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_134, mul_84, mul_86], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf314, buf313, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_7], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, primals_474, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf316 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_8, add_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf303, buf315, primals_475, primals_476, primals_477, primals_478, buf316, 11264, grid=grid(11264), stream=stream0)
        del primals_478
        # Topologically Sorted Source Nodes: [features_32_conv_0], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_479, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf318 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf319 = buf318; del buf318  # reuse
        # Topologically Sorted Source Nodes: [features_32_conv_1, sigmoid_139, mul_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf319, buf317, primals_480, primals_481, primals_482, primals_483, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_3], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_484, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf320, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf321 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf320, primals_485, primals_486, primals_487, primals_488, buf321, 67584, grid=grid(67584), stream=stream0)
        buf322 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf323 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_141, mul_88, features_32_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf323, buf321, 4224, 16, grid=grid(4224), stream=stream0)
        buf324 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_490, reinterpret_tensor(buf323, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_489, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf324)
        del primals_490
        buf325 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_143, mul_89], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf324, buf325, 192, grid=grid(192), stream=stream0)
        buf326 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_492, buf325, reinterpret_tensor(primals_491, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf326)
        del primals_492
        buf327 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_141, mul_88, mul_90], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf327, buf326, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_7], Original ATen: [aten.convolution]
        buf328 = extern_kernels.convolution(buf327, primals_493, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf329 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_8, add_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf316, buf328, primals_494, primals_495, primals_496, primals_497, buf329, 11264, grid=grid(11264), stream=stream0)
        del primals_497
        # Topologically Sorted Source Nodes: [features_33_conv_0], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_498, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf331 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf332 = buf331; del buf331  # reuse
        # Topologically Sorted Source Nodes: [features_33_conv_1, sigmoid_146, mul_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf332, buf330, primals_499, primals_500, primals_501, primals_502, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_3], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_503, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf333, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf334 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf333, primals_504, primals_505, primals_506, primals_507, buf334, 67584, grid=grid(67584), stream=stream0)
        buf335 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf336 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_148, mul_92, features_33_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf336, buf334, 4224, 16, grid=grid(4224), stream=stream0)
        buf337 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_509, reinterpret_tensor(buf336, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_508, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf337)
        del primals_509
        buf338 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_150, mul_93], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf337, buf338, 192, grid=grid(192), stream=stream0)
        buf339 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_511, buf338, reinterpret_tensor(primals_510, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf339)
        del primals_511
        buf340 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_148, mul_92, mul_94], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf340, buf339, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_7], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf341, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf342 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_8, add_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf329, buf341, primals_513, primals_514, primals_515, primals_516, buf342, 11264, grid=grid(11264), stream=stream0)
        del primals_516
        # Topologically Sorted Source Nodes: [features_34_conv_0], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, primals_517, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf344 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf345 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [features_34_conv_1, sigmoid_153, mul_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf345, buf343, primals_518, primals_519, primals_520, primals_521, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_3], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_522, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf346, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf347 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf346, primals_523, primals_524, primals_525, primals_526, buf347, 67584, grid=grid(67584), stream=stream0)
        buf348 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        buf349 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_155, mul_96, features_34_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf349, buf347, 4224, 16, grid=grid(4224), stream=stream0)
        buf350 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_528, reinterpret_tensor(buf349, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_527, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf350)
        del primals_528
        buf351 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_157, mul_97], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf350, buf351, 192, grid=grid(192), stream=stream0)
        buf352 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_530, buf351, reinterpret_tensor(primals_529, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf352)
        del primals_530
        buf353 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_155, mul_96, mul_98], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf353, buf352, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_7], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_531, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 176, 4, 4), (2816, 1, 704, 176))
        buf355 = empty_strided_cuda((4, 176, 4, 4), (2816, 1, 704, 176), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_8, add_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf342, buf354, primals_532, primals_533, primals_534, primals_535, buf355, 11264, grid=grid(11264), stream=stream0)
        del primals_535
        # Topologically Sorted Source Nodes: [features_35_conv_0], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_536, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 1056, 4, 4), (16896, 1, 4224, 1056))
        buf357 = empty_strided_cuda((4, 1056, 4, 4), (16896, 1, 4224, 1056), torch.float32)
        buf358 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [features_35_conv_1, sigmoid_160, mul_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf358, buf356, primals_537, primals_538, primals_539, primals_540, 67584, grid=grid(67584), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_3], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_541, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1056, bias=None)
        assert_size_stride(buf359, (4, 1056, 2, 2), (4224, 1, 2112, 1056))
        buf360 = empty_strided_cuda((4, 1056, 2, 2), (4224, 1, 2112, 1056), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf359, primals_542, primals_543, primals_544, primals_545, buf360, 16896, grid=grid(16896), stream=stream0)
        buf361 = empty_strided_cuda((4, 1056, 1, 1), (1056, 1, 4224, 4224), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_162, mul_100, features_35_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_39.run(buf360, buf361, 4224, grid=grid(4224), stream=stream0)
        buf362 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_547, reinterpret_tensor(buf361, (4, 1056), (1056, 1), 0), reinterpret_tensor(primals_546, (1056, 48), (1, 1056), 0), alpha=1, beta=1, out=buf362)
        del primals_547
        buf363 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_164, mul_101], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf362, buf363, 192, grid=grid(192), stream=stream0)
        buf364 = empty_strided_cuda((4, 1056), (1056, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_549, buf363, reinterpret_tensor(primals_548, (48, 1056), (1, 48), 0), alpha=1, beta=1, out=buf364)
        del primals_549
        buf365 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_162, mul_100, mul_102], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_40.run(buf365, buf364, 16896, grid=grid(16896), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_7], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_550, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf367 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf366, primals_551, primals_552, primals_553, primals_554, buf367, 4864, grid=grid(4864), stream=stream0)
        del primals_554
        # Topologically Sorted Source Nodes: [features_36_conv_0], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, primals_555, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf369 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf370 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [features_36_conv_1, sigmoid_167, mul_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf370, buf368, primals_556, primals_557, primals_558, primals_559, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_3], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_560, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf371, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf372 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf371, primals_561, primals_562, primals_563, primals_564, buf372, 29184, grid=grid(29184), stream=stream0)
        buf373 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_169, mul_104, features_36_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf372, buf373, 7296, grid=grid(7296), stream=stream0)
        buf374 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_566, reinterpret_tensor(buf373, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_565, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf374)
        del primals_566
        buf375 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_171, mul_105], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf374, buf375, 320, grid=grid(320), stream=stream0)
        buf376 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_568, buf375, reinterpret_tensor(primals_567, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf376)
        del primals_568
        buf377 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_169, mul_104, mul_106], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf377, buf376, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_7], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_569, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf379 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_8, add_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf367, buf378, primals_570, primals_571, primals_572, primals_573, buf379, 4864, grid=grid(4864), stream=stream0)
        del primals_573
        # Topologically Sorted Source Nodes: [features_37_conv_0], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_574, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf381 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf382 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [features_37_conv_1, sigmoid_174, mul_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf382, buf380, primals_575, primals_576, primals_577, primals_578, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_3], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_579, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf383, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf384 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf383, primals_580, primals_581, primals_582, primals_583, buf384, 29184, grid=grid(29184), stream=stream0)
        buf385 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_176, mul_108, features_37_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf384, buf385, 7296, grid=grid(7296), stream=stream0)
        buf386 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_585, reinterpret_tensor(buf385, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_584, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf386)
        del primals_585
        buf387 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_178, mul_109], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf386, buf387, 320, grid=grid(320), stream=stream0)
        buf388 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_587, buf387, reinterpret_tensor(primals_586, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf388)
        del primals_587
        buf389 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_176, mul_108, mul_110], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf389, buf388, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_7], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, primals_588, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf391 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_8, add_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf379, buf390, primals_589, primals_590, primals_591, primals_592, buf391, 4864, grid=grid(4864), stream=stream0)
        del primals_592
        # Topologically Sorted Source Nodes: [features_38_conv_0], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_593, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf393 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf394 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [features_38_conv_1, sigmoid_181, mul_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf394, buf392, primals_594, primals_595, primals_596, primals_597, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_3], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_598, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf395, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf396 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf395, primals_599, primals_600, primals_601, primals_602, buf396, 29184, grid=grid(29184), stream=stream0)
        buf397 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_183, mul_112, features_38_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf396, buf397, 7296, grid=grid(7296), stream=stream0)
        buf398 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_604, reinterpret_tensor(buf397, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_603, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf398)
        del primals_604
        buf399 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_185, mul_113], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf398, buf399, 320, grid=grid(320), stream=stream0)
        buf400 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_606, buf399, reinterpret_tensor(primals_605, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf400)
        del primals_606
        buf401 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_183, mul_112, mul_114], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf401, buf400, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_7], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_607, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf403 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_8, add_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf391, buf402, primals_608, primals_609, primals_610, primals_611, buf403, 4864, grid=grid(4864), stream=stream0)
        del primals_611
        # Topologically Sorted Source Nodes: [features_39_conv_0], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf405 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf406 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [features_39_conv_1, sigmoid_188, mul_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf406, buf404, primals_613, primals_614, primals_615, primals_616, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_3], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, primals_617, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf407, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf408 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf407, primals_618, primals_619, primals_620, primals_621, buf408, 29184, grid=grid(29184), stream=stream0)
        buf409 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_190, mul_116, features_39_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf408, buf409, 7296, grid=grid(7296), stream=stream0)
        buf410 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_623, reinterpret_tensor(buf409, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_622, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf410)
        del primals_623
        buf411 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_192, mul_117], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf410, buf411, 320, grid=grid(320), stream=stream0)
        buf412 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_625, buf411, reinterpret_tensor(primals_624, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf412)
        del primals_625
        buf413 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_190, mul_116, mul_118], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf413, buf412, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_7], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_626, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf415 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_8, add_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf403, buf414, primals_627, primals_628, primals_629, primals_630, buf415, 4864, grid=grid(4864), stream=stream0)
        del primals_630
        # Topologically Sorted Source Nodes: [features_40_conv_0], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_631, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf417 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf418 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [features_40_conv_1, sigmoid_195, mul_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf418, buf416, primals_632, primals_633, primals_634, primals_635, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_3], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_636, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf419, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf420 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf419, primals_637, primals_638, primals_639, primals_640, buf420, 29184, grid=grid(29184), stream=stream0)
        buf421 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_197, mul_120, features_40_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf420, buf421, 7296, grid=grid(7296), stream=stream0)
        buf422 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_642, reinterpret_tensor(buf421, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_641, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf422)
        del primals_642
        buf423 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_199, mul_121], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf422, buf423, 320, grid=grid(320), stream=stream0)
        buf424 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_644, buf423, reinterpret_tensor(primals_643, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf424)
        del primals_644
        buf425 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_197, mul_120, mul_122], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf425, buf424, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_7], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_645, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf427 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_8, add_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf415, buf426, primals_646, primals_647, primals_648, primals_649, buf427, 4864, grid=grid(4864), stream=stream0)
        del primals_649
        # Topologically Sorted Source Nodes: [features_41_conv_0], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_650, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf429 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf430 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [features_41_conv_1, sigmoid_202, mul_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf430, buf428, primals_651, primals_652, primals_653, primals_654, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_41_conv_3], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_655, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf431, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf432 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf431, primals_656, primals_657, primals_658, primals_659, buf432, 29184, grid=grid(29184), stream=stream0)
        buf433 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_204, mul_124, features_41_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf432, buf433, 7296, grid=grid(7296), stream=stream0)
        buf434 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_661, reinterpret_tensor(buf433, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_660, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf434)
        del primals_661
        buf435 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_206, mul_125], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf434, buf435, 320, grid=grid(320), stream=stream0)
        buf436 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_663, buf435, reinterpret_tensor(primals_662, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf436)
        del primals_663
        buf437 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_204, mul_124, mul_126], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf437, buf436, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_41_conv_7], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_664, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf439 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_8, add_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf427, buf438, primals_665, primals_666, primals_667, primals_668, buf439, 4864, grid=grid(4864), stream=stream0)
        del primals_668
        # Topologically Sorted Source Nodes: [features_42_conv_0], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, primals_669, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf441 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf442 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [features_42_conv_1, sigmoid_209, mul_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf442, buf440, primals_670, primals_671, primals_672, primals_673, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_42_conv_3], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_674, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf443, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf444 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf443, primals_675, primals_676, primals_677, primals_678, buf444, 29184, grid=grid(29184), stream=stream0)
        buf445 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_211, mul_128, features_42_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf444, buf445, 7296, grid=grid(7296), stream=stream0)
        buf446 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_680, reinterpret_tensor(buf445, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_679, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf446)
        del primals_680
        buf447 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_213, mul_129], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf446, buf447, 320, grid=grid(320), stream=stream0)
        buf448 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_682, buf447, reinterpret_tensor(primals_681, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf448)
        del primals_682
        buf449 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_211, mul_128, mul_130], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf449, buf448, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_42_conv_7], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, primals_683, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf451 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_8, add_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf439, buf450, primals_684, primals_685, primals_686, primals_687, buf451, 4864, grid=grid(4864), stream=stream0)
        del primals_687
        # Topologically Sorted Source Nodes: [features_43_conv_0], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, primals_688, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf453 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf454 = buf453; del buf453  # reuse
        # Topologically Sorted Source Nodes: [features_43_conv_1, sigmoid_216, mul_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf454, buf452, primals_689, primals_690, primals_691, primals_692, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_43_conv_3], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf454, primals_693, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf455, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf456 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf455, primals_694, primals_695, primals_696, primals_697, buf456, 29184, grid=grid(29184), stream=stream0)
        buf457 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_218, mul_132, features_43_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf456, buf457, 7296, grid=grid(7296), stream=stream0)
        buf458 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_699, reinterpret_tensor(buf457, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_698, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf458)
        del primals_699
        buf459 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_220, mul_133], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf458, buf459, 320, grid=grid(320), stream=stream0)
        buf460 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_701, buf459, reinterpret_tensor(primals_700, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf460)
        del primals_701
        buf461 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_218, mul_132, mul_134], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf461, buf460, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_43_conv_7], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_702, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf463 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_8, add_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf451, buf462, primals_703, primals_704, primals_705, primals_706, buf463, 4864, grid=grid(4864), stream=stream0)
        del primals_706
        # Topologically Sorted Source Nodes: [features_44_conv_0], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_707, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf465 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf466 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [features_44_conv_1, sigmoid_223, mul_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf466, buf464, primals_708, primals_709, primals_710, primals_711, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_44_conv_3], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_712, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf467, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf468 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf467, primals_713, primals_714, primals_715, primals_716, buf468, 29184, grid=grid(29184), stream=stream0)
        buf469 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_225, mul_136, features_44_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf468, buf469, 7296, grid=grid(7296), stream=stream0)
        buf470 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_718, reinterpret_tensor(buf469, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_717, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf470)
        del primals_718
        buf471 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_227, mul_137], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf470, buf471, 320, grid=grid(320), stream=stream0)
        buf472 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_720, buf471, reinterpret_tensor(primals_719, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf472)
        del primals_720
        buf473 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_225, mul_136, mul_138], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf473, buf472, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_44_conv_7], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, primals_721, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf475 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_8, add_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf463, buf474, primals_722, primals_723, primals_724, primals_725, buf475, 4864, grid=grid(4864), stream=stream0)
        del primals_725
        # Topologically Sorted Source Nodes: [features_45_conv_0], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_726, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf477 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf478 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [features_45_conv_1, sigmoid_230, mul_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf478, buf476, primals_727, primals_728, primals_729, primals_730, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_45_conv_3], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_731, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf479, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf480 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf479, primals_732, primals_733, primals_734, primals_735, buf480, 29184, grid=grid(29184), stream=stream0)
        buf481 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_232, mul_140, features_45_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf480, buf481, 7296, grid=grid(7296), stream=stream0)
        buf482 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_737, reinterpret_tensor(buf481, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_736, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf482)
        del primals_737
        buf483 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_234, mul_141], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf482, buf483, 320, grid=grid(320), stream=stream0)
        buf484 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_739, buf483, reinterpret_tensor(primals_738, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf484)
        del primals_739
        buf485 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_232, mul_140, mul_142], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf485, buf484, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_45_conv_7], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_740, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf487 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_8, add_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf475, buf486, primals_741, primals_742, primals_743, primals_744, buf487, 4864, grid=grid(4864), stream=stream0)
        del primals_744
        # Topologically Sorted Source Nodes: [features_46_conv_0], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf487, primals_745, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf489 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf490 = buf489; del buf489  # reuse
        # Topologically Sorted Source Nodes: [features_46_conv_1, sigmoid_237, mul_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf490, buf488, primals_746, primals_747, primals_748, primals_749, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_46_conv_3], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, primals_750, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf491, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf492 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf491, primals_751, primals_752, primals_753, primals_754, buf492, 29184, grid=grid(29184), stream=stream0)
        buf493 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_239, mul_144, features_46_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf492, buf493, 7296, grid=grid(7296), stream=stream0)
        buf494 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_756, reinterpret_tensor(buf493, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_755, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf494)
        del primals_756
        buf495 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_241, mul_145], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf494, buf495, 320, grid=grid(320), stream=stream0)
        buf496 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_758, buf495, reinterpret_tensor(primals_757, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf496)
        del primals_758
        buf497 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_239, mul_144, mul_146], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf497, buf496, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_46_conv_7], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_759, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf499 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_8, add_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf487, buf498, primals_760, primals_761, primals_762, primals_763, buf499, 4864, grid=grid(4864), stream=stream0)
        del primals_763
        # Topologically Sorted Source Nodes: [features_47_conv_0], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, primals_764, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf501 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf502 = buf501; del buf501  # reuse
        # Topologically Sorted Source Nodes: [features_47_conv_1, sigmoid_244, mul_147], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf502, buf500, primals_765, primals_766, primals_767, primals_768, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_47_conv_3], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, primals_769, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf503, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf504 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf503, primals_770, primals_771, primals_772, primals_773, buf504, 29184, grid=grid(29184), stream=stream0)
        buf505 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_246, mul_148, features_47_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf504, buf505, 7296, grid=grid(7296), stream=stream0)
        buf506 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_775, reinterpret_tensor(buf505, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_774, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf506)
        del primals_775
        buf507 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_248, mul_149], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf506, buf507, 320, grid=grid(320), stream=stream0)
        buf508 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_777, buf507, reinterpret_tensor(primals_776, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf508)
        del primals_777
        buf509 = buf504; del buf504  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_246, mul_148, mul_150], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf509, buf508, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_47_conv_7], Original ATen: [aten.convolution]
        buf510 = extern_kernels.convolution(buf509, primals_778, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf510, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf511 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_8, add_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf499, buf510, primals_779, primals_780, primals_781, primals_782, buf511, 4864, grid=grid(4864), stream=stream0)
        del primals_782
        # Topologically Sorted Source Nodes: [features_48_conv_0], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_783, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf513 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf514 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [features_48_conv_1, sigmoid_251, mul_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf514, buf512, primals_784, primals_785, primals_786, primals_787, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_48_conv_3], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_788, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf515, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf516 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf515, primals_789, primals_790, primals_791, primals_792, buf516, 29184, grid=grid(29184), stream=stream0)
        buf517 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_253, mul_152, features_48_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf516, buf517, 7296, grid=grid(7296), stream=stream0)
        buf518 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_794, reinterpret_tensor(buf517, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_793, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf518)
        del primals_794
        buf519 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_255, mul_153], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf518, buf519, 320, grid=grid(320), stream=stream0)
        buf520 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_796, buf519, reinterpret_tensor(primals_795, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf520)
        del primals_796
        buf521 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_253, mul_152, mul_154], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf521, buf520, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_48_conv_7], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_797, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf523 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_8, add_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf511, buf522, primals_798, primals_799, primals_800, primals_801, buf523, 4864, grid=grid(4864), stream=stream0)
        del primals_801
        # Topologically Sorted Source Nodes: [features_49_conv_0], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, primals_802, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf525 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf526 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [features_49_conv_1, sigmoid_258, mul_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf526, buf524, primals_803, primals_804, primals_805, primals_806, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_49_conv_3], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, primals_807, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf527, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf528 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf527, primals_808, primals_809, primals_810, primals_811, buf528, 29184, grid=grid(29184), stream=stream0)
        buf529 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_260, mul_156, features_49_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf528, buf529, 7296, grid=grid(7296), stream=stream0)
        buf530 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_813, reinterpret_tensor(buf529, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_812, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf530)
        del primals_813
        buf531 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_262, mul_157], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf530, buf531, 320, grid=grid(320), stream=stream0)
        buf532 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_815, buf531, reinterpret_tensor(primals_814, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf532)
        del primals_815
        buf533 = buf528; del buf528  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_260, mul_156, mul_158], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf533, buf532, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_49_conv_7], Original ATen: [aten.convolution]
        buf534 = extern_kernels.convolution(buf533, primals_816, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf534, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf535 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_8, add_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf523, buf534, primals_817, primals_818, primals_819, primals_820, buf535, 4864, grid=grid(4864), stream=stream0)
        del primals_820
        # Topologically Sorted Source Nodes: [features_50_conv_0], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_821, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf537 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf538 = buf537; del buf537  # reuse
        # Topologically Sorted Source Nodes: [features_50_conv_1, sigmoid_265, mul_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf538, buf536, primals_822, primals_823, primals_824, primals_825, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_50_conv_3], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_826, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf539, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf540 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf539, primals_827, primals_828, primals_829, primals_830, buf540, 29184, grid=grid(29184), stream=stream0)
        buf541 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_267, mul_160, features_50_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf540, buf541, 7296, grid=grid(7296), stream=stream0)
        buf542 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_832, reinterpret_tensor(buf541, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_831, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf542)
        del primals_832
        buf543 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_269, mul_161], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf542, buf543, 320, grid=grid(320), stream=stream0)
        buf544 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_834, buf543, reinterpret_tensor(primals_833, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf544)
        del primals_834
        buf545 = buf540; del buf540  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_267, mul_160, mul_162], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf545, buf544, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_50_conv_7], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, primals_835, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf547 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_8, add_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf535, buf546, primals_836, primals_837, primals_838, primals_839, buf547, 4864, grid=grid(4864), stream=stream0)
        del primals_839
        # Topologically Sorted Source Nodes: [features_51_conv_0], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, primals_840, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf549 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf550 = buf549; del buf549  # reuse
        # Topologically Sorted Source Nodes: [features_51_conv_1, sigmoid_272, mul_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf550, buf548, primals_841, primals_842, primals_843, primals_844, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_51_conv_3], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_845, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf551, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf552 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf551, primals_846, primals_847, primals_848, primals_849, buf552, 29184, grid=grid(29184), stream=stream0)
        buf553 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_274, mul_164, features_51_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf552, buf553, 7296, grid=grid(7296), stream=stream0)
        buf554 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_851, reinterpret_tensor(buf553, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_850, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf554)
        del primals_851
        buf555 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_276, mul_165], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf554, buf555, 320, grid=grid(320), stream=stream0)
        buf556 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_853, buf555, reinterpret_tensor(primals_852, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf556)
        del primals_853
        buf557 = buf552; del buf552  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_274, mul_164, mul_166], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf557, buf556, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_51_conv_7], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, primals_854, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf558, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf559 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_8, add_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf547, buf558, primals_855, primals_856, primals_857, primals_858, buf559, 4864, grid=grid(4864), stream=stream0)
        del primals_858
        # Topologically Sorted Source Nodes: [features_52_conv_0], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_859, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf561 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf562 = buf561; del buf561  # reuse
        # Topologically Sorted Source Nodes: [features_52_conv_1, sigmoid_279, mul_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf562, buf560, primals_860, primals_861, primals_862, primals_863, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_52_conv_3], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, primals_864, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf563, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf564 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf563, primals_865, primals_866, primals_867, primals_868, buf564, 29184, grid=grid(29184), stream=stream0)
        buf565 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_281, mul_168, features_52_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf564, buf565, 7296, grid=grid(7296), stream=stream0)
        buf566 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_870, reinterpret_tensor(buf565, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_869, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf566)
        del primals_870
        buf567 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_283, mul_169], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf566, buf567, 320, grid=grid(320), stream=stream0)
        buf568 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_872, buf567, reinterpret_tensor(primals_871, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf568)
        del primals_872
        buf569 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_281, mul_168, mul_170], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf569, buf568, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_52_conv_7], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_873, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf570, (4, 304, 2, 2), (1216, 1, 608, 304))
        buf571 = empty_strided_cuda((4, 304, 2, 2), (1216, 1, 608, 304), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_8, add_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf559, buf570, primals_874, primals_875, primals_876, primals_877, buf571, 4864, grid=grid(4864), stream=stream0)
        del primals_877
        # Topologically Sorted Source Nodes: [features_53_conv_0], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_878, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf573 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        buf574 = buf573; del buf573  # reuse
        # Topologically Sorted Source Nodes: [features_53_conv_1, sigmoid_286, mul_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf574, buf572, primals_879, primals_880, primals_881, primals_882, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_53_conv_3], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf574, primals_883, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1824, bias=None)
        assert_size_stride(buf575, (4, 1824, 2, 2), (7296, 1, 3648, 1824))
        buf576 = empty_strided_cuda((4, 1824, 2, 2), (7296, 1, 3648, 1824), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf575, primals_884, primals_885, primals_886, primals_887, buf576, 29184, grid=grid(29184), stream=stream0)
        buf577 = empty_strided_cuda((4, 1824, 1, 1), (1824, 1, 7296, 7296), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_288, mul_172, features_53_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf576, buf577, 7296, grid=grid(7296), stream=stream0)
        buf578 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_889, reinterpret_tensor(buf577, (4, 1824), (1824, 1), 0), reinterpret_tensor(primals_888, (1824, 80), (1, 1824), 0), alpha=1, beta=1, out=buf578)
        del primals_889
        buf579 = empty_strided_cuda((4, 80), (80, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_290, mul_173], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf578, buf579, 320, grid=grid(320), stream=stream0)
        buf580 = empty_strided_cuda((4, 1824), (1824, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_891, buf579, reinterpret_tensor(primals_890, (80, 1824), (1, 80), 0), alpha=1, beta=1, out=buf580)
        del primals_891
        buf581 = buf576; del buf576  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_288, mul_172, mul_174], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf581, buf580, 29184, grid=grid(29184), stream=stream0)
        # Topologically Sorted Source Nodes: [features_53_conv_7], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, primals_892, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf583 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf582, primals_893, primals_894, primals_895, primals_896, buf583, 8192, grid=grid(8192), stream=stream0)
        del primals_896
        # Topologically Sorted Source Nodes: [features_54_conv_0], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf583, primals_897, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf585 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf586 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [features_54_conv_1, sigmoid_293, mul_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49.run(buf586, buf584, primals_898, primals_899, primals_900, primals_901, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_54_conv_3], Original ATen: [aten.convolution]
        buf587 = extern_kernels.convolution(buf586, primals_902, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf587, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf588 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf587, primals_903, primals_904, primals_905, primals_906, buf588, 49152, grid=grid(49152), stream=stream0)
        buf589 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_295, mul_176, features_54_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_51.run(buf588, buf589, 12288, grid=grid(12288), stream=stream0)
        buf590 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_908, reinterpret_tensor(buf589, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_907, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf590)
        del primals_908
        buf591 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_297, mul_177], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_52.run(buf590, buf591, 512, grid=grid(512), stream=stream0)
        buf592 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_910, buf591, reinterpret_tensor(primals_909, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf592)
        del primals_910
        buf593 = buf588; del buf588  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_295, mul_176, mul_178], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_53.run(buf593, buf592, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_54_conv_7], Original ATen: [aten.convolution]
        buf594 = extern_kernels.convolution(buf593, primals_911, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf594, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf595 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_8, add_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_54.run(buf583, buf594, primals_912, primals_913, primals_914, primals_915, buf595, 8192, grid=grid(8192), stream=stream0)
        del primals_915
        # Topologically Sorted Source Nodes: [features_55_conv_0], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_916, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf597 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf598 = buf597; del buf597  # reuse
        # Topologically Sorted Source Nodes: [features_55_conv_1, sigmoid_300, mul_179], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49.run(buf598, buf596, primals_917, primals_918, primals_919, primals_920, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_55_conv_3], Original ATen: [aten.convolution]
        buf599 = extern_kernels.convolution(buf598, primals_921, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf599, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf600 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf599, primals_922, primals_923, primals_924, primals_925, buf600, 49152, grid=grid(49152), stream=stream0)
        buf601 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_302, mul_180, features_55_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_51.run(buf600, buf601, 12288, grid=grid(12288), stream=stream0)
        buf602 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_927, reinterpret_tensor(buf601, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_926, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf602)
        del primals_927
        buf603 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_304, mul_181], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_52.run(buf602, buf603, 512, grid=grid(512), stream=stream0)
        buf604 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_929, buf603, reinterpret_tensor(primals_928, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf604)
        del primals_929
        buf605 = buf600; del buf600  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_302, mul_180, mul_182], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_53.run(buf605, buf604, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_55_conv_7], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_930, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf607 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_8, add_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_54.run(buf595, buf606, primals_931, primals_932, primals_933, primals_934, buf607, 8192, grid=grid(8192), stream=stream0)
        del primals_934
        # Topologically Sorted Source Nodes: [features_56_conv_0], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(buf607, primals_935, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf609 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf610 = buf609; del buf609  # reuse
        # Topologically Sorted Source Nodes: [features_56_conv_1, sigmoid_307, mul_183], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49.run(buf610, buf608, primals_936, primals_937, primals_938, primals_939, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_56_conv_3], Original ATen: [aten.convolution]
        buf611 = extern_kernels.convolution(buf610, primals_940, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf611, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf612 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf611, primals_941, primals_942, primals_943, primals_944, buf612, 49152, grid=grid(49152), stream=stream0)
        buf613 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_309, mul_184, features_56_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_51.run(buf612, buf613, 12288, grid=grid(12288), stream=stream0)
        buf614 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_946, reinterpret_tensor(buf613, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_945, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf614)
        del primals_946
        buf615 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_311, mul_185], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_52.run(buf614, buf615, 512, grid=grid(512), stream=stream0)
        buf616 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_948, buf615, reinterpret_tensor(primals_947, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf616)
        del primals_948
        buf617 = buf612; del buf612  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_309, mul_184, mul_186], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_53.run(buf617, buf616, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_56_conv_7], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, primals_949, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf618, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf619 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_8, add_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_54.run(buf607, buf618, primals_950, primals_951, primals_952, primals_953, buf619, 8192, grid=grid(8192), stream=stream0)
        del primals_953
        # Topologically Sorted Source Nodes: [features_57_conv_0], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_954, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf621 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf622 = buf621; del buf621  # reuse
        # Topologically Sorted Source Nodes: [features_57_conv_1, sigmoid_314, mul_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_49.run(buf622, buf620, primals_955, primals_956, primals_957, primals_958, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_57_conv_3], Original ATen: [aten.convolution]
        buf623 = extern_kernels.convolution(buf622, primals_959, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf623, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf624 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_50.run(buf623, primals_960, primals_961, primals_962, primals_963, buf624, 49152, grid=grid(49152), stream=stream0)
        buf625 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_316, mul_188, features_57_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_51.run(buf624, buf625, 12288, grid=grid(12288), stream=stream0)
        buf626 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_965, reinterpret_tensor(buf625, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_964, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf626)
        del primals_965
        buf627 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_318, mul_189], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_52.run(buf626, buf627, 512, grid=grid(512), stream=stream0)
        buf628 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_967, buf627, reinterpret_tensor(primals_966, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf628)
        del primals_967
        buf629 = buf624; del buf624  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_316, mul_188, mul_190], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_53.run(buf629, buf628, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_57_conv_7], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, primals_968, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf631 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_8, add_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_54.run(buf619, buf630, primals_969, primals_970, primals_971, primals_972, buf631, 8192, grid=grid(8192), stream=stream0)
        del primals_972
        # Topologically Sorted Source Nodes: [conv_0], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_973, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (4, 1792, 2, 2), (7168, 1, 3584, 1792))
        buf633 = empty_strided_cuda((4, 1792, 2, 2), (7168, 1, 3584, 1792), torch.float32)
        # Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_55.run(buf632, primals_974, primals_975, primals_976, primals_977, buf633, 28672, grid=grid(28672), stream=stream0)
        buf634 = empty_strided_cuda((4, 1792, 1, 1), (1792, 1, 7168, 7168), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_321, mul_191, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_56.run(buf633, buf634, 7168, grid=grid(7168), stream=stream0)
        del buf633
        buf635 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [classifier], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_979, reinterpret_tensor(buf634, (4, 1792), (1792, 1), 0), reinterpret_tensor(primals_978, (1792, 1000), (1, 1792), 0), alpha=1, beta=1, out=buf635)
        del primals_979
    return (buf635, buf0, buf1, primals_3, primals_4, primals_5, primals_6, buf2, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, buf3, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, buf4, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, buf5, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, buf6, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, buf7, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, buf8, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, buf9, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, buf10, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, buf11, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, buf12, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, buf13, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, buf14, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_151, primals_152, primals_153, primals_154, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_170, primals_171, primals_172, primals_173, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_189, primals_190, primals_191, primals_192, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_208, primals_209, primals_210, primals_211, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_246, primals_247, primals_248, primals_249, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_265, primals_266, primals_267, primals_268, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_284, primals_285, primals_286, primals_287, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_303, primals_304, primals_305, primals_306, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_341, primals_342, primals_343, primals_344, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_360, primals_361, primals_362, primals_363, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_379, primals_380, primals_381, primals_382, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_398, primals_399, primals_400, primals_401, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_436, primals_437, primals_438, primals_439, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_455, primals_456, primals_457, primals_458, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_474, primals_475, primals_476, primals_477, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_493, primals_494, primals_495, primals_496, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_512, primals_513, primals_514, primals_515, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_531, primals_532, primals_533, primals_534, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_550, primals_551, primals_552, primals_553, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_569, primals_570, primals_571, primals_572, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_588, primals_589, primals_590, primals_591, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_607, primals_608, primals_609, primals_610, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_626, primals_627, primals_628, primals_629, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_645, primals_646, primals_647, primals_648, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_664, primals_665, primals_666, primals_667, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_683, primals_684, primals_685, primals_686, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_702, primals_703, primals_704, primals_705, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_721, primals_722, primals_723, primals_724, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_740, primals_741, primals_742, primals_743, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_759, primals_760, primals_761, primals_762, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_778, primals_779, primals_780, primals_781, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_797, primals_798, primals_799, primals_800, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_816, primals_817, primals_818, primals_819, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_835, primals_836, primals_837, primals_838, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_854, primals_855, primals_856, primals_857, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_873, primals_874, primals_875, primals_876, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_892, primals_893, primals_894, primals_895, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_911, primals_912, primals_913, primals_914, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_930, primals_931, primals_932, primals_933, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_949, primals_950, primals_951, primals_952, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_968, primals_969, primals_970, primals_971, primals_973, primals_974, primals_975, primals_976, primals_977, buf15, buf17, buf18, buf20, buf21, buf22, buf23, buf25, buf26, buf27, buf28, buf30, buf31, buf32, buf33, buf35, buf36, buf37, buf38, buf40, buf41, buf42, buf43, buf45, buf46, buf47, buf48, buf50, buf51, buf52, buf53, buf55, buf56, buf57, buf58, buf60, buf61, buf62, buf63, buf65, buf66, buf67, buf68, buf70, buf71, buf72, buf73, buf75, buf76, buf77, buf78, buf80, buf81, buf82, buf83, buf85, buf86, reinterpret_tensor(buf89, (4, 320), (320, 1), 0), buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf98, buf99, reinterpret_tensor(buf102, (4, 640), (640, 1), 0), buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf111, buf112, reinterpret_tensor(buf115, (4, 640), (640, 1), 0), buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf124, buf125, reinterpret_tensor(buf128, (4, 640), (640, 1), 0), buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf137, buf138, reinterpret_tensor(buf141, (4, 640), (640, 1), 0), buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf150, buf151, reinterpret_tensor(buf154, (4, 640), (640, 1), 0), buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf163, buf164, reinterpret_tensor(buf167, (4, 640), (640, 1), 0), buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf176, buf177, reinterpret_tensor(buf180, (4, 960), (960, 1), 0), buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf189, buf190, reinterpret_tensor(buf193, (4, 1056), (1056, 1), 0), buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf202, buf203, reinterpret_tensor(buf206, (4, 1056), (1056, 1), 0), buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf215, buf216, reinterpret_tensor(buf219, (4, 1056), (1056, 1), 0), buf220, buf221, buf222, buf223, buf224, buf225, buf226, buf228, buf229, reinterpret_tensor(buf232, (4, 1056), (1056, 1), 0), buf233, buf234, buf235, buf236, buf237, buf238, buf239, buf241, buf242, reinterpret_tensor(buf245, (4, 1056), (1056, 1), 0), buf246, buf247, buf248, buf249, buf250, buf251, buf252, buf254, buf255, reinterpret_tensor(buf258, (4, 1056), (1056, 1), 0), buf259, buf260, buf261, buf262, buf263, buf264, buf265, buf267, buf268, reinterpret_tensor(buf271, (4, 1056), (1056, 1), 0), buf272, buf273, buf274, buf275, buf276, buf277, buf278, buf280, buf281, reinterpret_tensor(buf284, (4, 1056), (1056, 1), 0), buf285, buf286, buf287, buf288, buf289, buf290, buf291, buf293, buf294, reinterpret_tensor(buf297, (4, 1056), (1056, 1), 0), buf298, buf299, buf300, buf301, buf302, buf303, buf304, buf306, buf307, reinterpret_tensor(buf310, (4, 1056), (1056, 1), 0), buf311, buf312, buf313, buf314, buf315, buf316, buf317, buf319, buf320, reinterpret_tensor(buf323, (4, 1056), (1056, 1), 0), buf324, buf325, buf326, buf327, buf328, buf329, buf330, buf332, buf333, reinterpret_tensor(buf336, (4, 1056), (1056, 1), 0), buf337, buf338, buf339, buf340, buf341, buf342, buf343, buf345, buf346, reinterpret_tensor(buf349, (4, 1056), (1056, 1), 0), buf350, buf351, buf352, buf353, buf354, buf355, buf356, buf358, buf359, reinterpret_tensor(buf361, (4, 1056), (1056, 1), 0), buf362, buf363, buf364, buf365, buf366, buf367, buf368, buf370, buf371, reinterpret_tensor(buf373, (4, 1824), (1824, 1), 0), buf374, buf375, buf376, buf377, buf378, buf379, buf380, buf382, buf383, reinterpret_tensor(buf385, (4, 1824), (1824, 1), 0), buf386, buf387, buf388, buf389, buf390, buf391, buf392, buf394, buf395, reinterpret_tensor(buf397, (4, 1824), (1824, 1), 0), buf398, buf399, buf400, buf401, buf402, buf403, buf404, buf406, buf407, reinterpret_tensor(buf409, (4, 1824), (1824, 1), 0), buf410, buf411, buf412, buf413, buf414, buf415, buf416, buf418, buf419, reinterpret_tensor(buf421, (4, 1824), (1824, 1), 0), buf422, buf423, buf424, buf425, buf426, buf427, buf428, buf430, buf431, reinterpret_tensor(buf433, (4, 1824), (1824, 1), 0), buf434, buf435, buf436, buf437, buf438, buf439, buf440, buf442, buf443, reinterpret_tensor(buf445, (4, 1824), (1824, 1), 0), buf446, buf447, buf448, buf449, buf450, buf451, buf452, buf454, buf455, reinterpret_tensor(buf457, (4, 1824), (1824, 1), 0), buf458, buf459, buf460, buf461, buf462, buf463, buf464, buf466, buf467, reinterpret_tensor(buf469, (4, 1824), (1824, 1), 0), buf470, buf471, buf472, buf473, buf474, buf475, buf476, buf478, buf479, reinterpret_tensor(buf481, (4, 1824), (1824, 1), 0), buf482, buf483, buf484, buf485, buf486, buf487, buf488, buf490, buf491, reinterpret_tensor(buf493, (4, 1824), (1824, 1), 0), buf494, buf495, buf496, buf497, buf498, buf499, buf500, buf502, buf503, reinterpret_tensor(buf505, (4, 1824), (1824, 1), 0), buf506, buf507, buf508, buf509, buf510, buf511, buf512, buf514, buf515, reinterpret_tensor(buf517, (4, 1824), (1824, 1), 0), buf518, buf519, buf520, buf521, buf522, buf523, buf524, buf526, buf527, reinterpret_tensor(buf529, (4, 1824), (1824, 1), 0), buf530, buf531, buf532, buf533, buf534, buf535, buf536, buf538, buf539, reinterpret_tensor(buf541, (4, 1824), (1824, 1), 0), buf542, buf543, buf544, buf545, buf546, buf547, buf548, buf550, buf551, reinterpret_tensor(buf553, (4, 1824), (1824, 1), 0), buf554, buf555, buf556, buf557, buf558, buf559, buf560, buf562, buf563, reinterpret_tensor(buf565, (4, 1824), (1824, 1), 0), buf566, buf567, buf568, buf569, buf570, buf571, buf572, buf574, buf575, reinterpret_tensor(buf577, (4, 1824), (1824, 1), 0), buf578, buf579, buf580, buf581, buf582, buf583, buf584, buf586, buf587, reinterpret_tensor(buf589, (4, 3072), (3072, 1), 0), buf590, buf591, buf592, buf593, buf594, buf595, buf596, buf598, buf599, reinterpret_tensor(buf601, (4, 3072), (3072, 1), 0), buf602, buf603, buf604, buf605, buf606, buf607, buf608, buf610, buf611, reinterpret_tensor(buf613, (4, 3072), (3072, 1), 0), buf614, buf615, buf616, buf617, buf618, buf619, buf620, buf622, buf623, reinterpret_tensor(buf625, (4, 3072), (3072, 1), 0), buf626, buf627, buf628, buf629, buf630, buf631, buf632, reinterpret_tensor(buf634, (4, 1792), (1792, 1), 0), primals_978, primals_966, primals_964, primals_947, primals_945, primals_928, primals_926, primals_909, primals_907, primals_890, primals_888, primals_871, primals_869, primals_852, primals_850, primals_833, primals_831, primals_814, primals_812, primals_795, primals_793, primals_776, primals_774, primals_757, primals_755, primals_738, primals_736, primals_719, primals_717, primals_700, primals_698, primals_681, primals_679, primals_662, primals_660, primals_643, primals_641, primals_624, primals_622, primals_605, primals_603, primals_586, primals_584, primals_567, primals_565, primals_548, primals_546, primals_529, primals_527, primals_510, primals_508, primals_491, primals_489, primals_472, primals_470, primals_453, primals_451, primals_434, primals_432, primals_415, primals_413, primals_396, primals_394, primals_377, primals_375, primals_358, primals_356, primals_339, primals_337, primals_320, primals_318, primals_301, primals_299, primals_282, primals_280, primals_263, primals_261, primals_244, primals_242, primals_225, primals_223, primals_206, primals_204, primals_187, primals_185, primals_168, primals_166, primals_149, primals_147, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((24, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((96, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((48, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((80, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((320, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((80, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((320, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((80, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((320, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((80, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((320, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((80, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((320, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((320, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((24, 320), (320, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((320, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((160, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((640, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((40, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((640, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((640, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((40, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((640, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((640, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((40, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((640, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((640, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((40, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((640, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((640, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((40, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((640, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((640, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((40, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((640, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((160, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((176, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((176, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((1056, 176, 1, 1), (176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((1056, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((48, 1056), (1056, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((1056, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((1056, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((304, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((304, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((1824, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((1824, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((80, 1824), (1824, 1), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((1824, 80), (80, 1), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((1824, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((512, 1824, 1, 1), (1824, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((1792, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((1000, 1792), (1792, 1), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
