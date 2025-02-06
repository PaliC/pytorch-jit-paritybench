# AOT ID: ['11_forward']
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


# kernel path: inductor_cache/5i/c5id5v6645j5a5swq6nwbwtyvuol6sty6sg3eajt4lk2mycblo24.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
# Topologically Sorted Source Nodes: [features_3_conv_1, sigmoid_4, mul_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_3_conv_1 => add_13, mul_19, mul_20, sub_5
#   mul_4 => mul_21
#   sigmoid_4 => sigmoid_3
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_45), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_47), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_13,), kwargs = {})
#   %mul_21 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_13, %sigmoid_3), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_3_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_3_conv_4 => add_15, mul_23, mul_24, sub_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_23, %unsqueeze_53), kwargs = {})
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, %unsqueeze_55), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_4_conv_1, sigmoid_5, mul_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_4_conv_1 => add_17, mul_26, mul_27, sub_7
#   mul_5 => mul_28
#   sigmoid_5 => sigmoid_4
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_26, %unsqueeze_61), kwargs = {})
#   %add_17 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_27, %unsqueeze_63), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_17,), kwargs = {})
#   %mul_28 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_17, %sigmoid_4), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_4_conv_4, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_3 => add_20
#   features_4_conv_4 => add_19, mul_30, mul_31, sub_8
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_30, %unsqueeze_69), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_31, %unsqueeze_71), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %add_19), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_7_conv_1, sigmoid_8, mul_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_7_conv_1 => add_32, mul_47, mul_48, sub_13
#   mul_8 => mul_49
#   sigmoid_8 => sigmoid_7
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_48 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_47, %unsqueeze_109), kwargs = {})
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_48, %unsqueeze_111), kwargs = {})
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_32,), kwargs = {})
#   %mul_49 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_32, %sigmoid_7), kwargs = {})
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


# kernel path: inductor_cache/45/c453isruos7ft3rq5joeqs4sgzzbwmr4d4yqosfwj6dolwssktbi.py
# Topologically Sorted Source Nodes: [features_7_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_7_conv_4 => add_34, mul_51, mul_52, sub_14
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %unsqueeze_117), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_52, %unsqueeze_119), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/u6/cu66eevj43pggorsmtsdmc5cyq34pissw673e2ctmuhweop5z47g.py
# Topologically Sorted Source Nodes: [features_8_conv_1, sigmoid_9, mul_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_8_conv_1 => add_36, mul_54, mul_55, sub_15
#   mul_9 => mul_56
#   sigmoid_9 => sigmoid_8
# Graph fragment:
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %unsqueeze_125), kwargs = {})
#   %add_36 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %unsqueeze_127), kwargs = {})
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_36,), kwargs = {})
#   %mul_56 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_36, %sigmoid_8), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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


# kernel path: inductor_cache/cb/ccb6bryhwsdfwmba3ubojmoeuf4zazwog5fxzdwgyhq2obgh7hr5.py
# Topologically Sorted Source Nodes: [features_8_conv_4, add_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_6 => add_39
#   features_8_conv_4 => add_38, mul_58, mul_59, sub_16
# Graph fragment:
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_129), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %unsqueeze_131), kwargs = {})
#   %mul_59 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_58, %unsqueeze_133), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_59, %unsqueeze_135), kwargs = {})
#   %add_39 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %add_38), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
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


# kernel path: inductor_cache/uj/cujmyv3b6ushhlnxfh62owdife4m4m3a2wn76pyezsnytimdqolh.py
# Topologically Sorted Source Nodes: [features_11_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_11_conv_4 => add_53, mul_79, mul_80, sub_22
# Graph fragment:
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_181), kwargs = {})
#   %add_53 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_183), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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


# kernel path: inductor_cache/bg/cbgupxie6wpwco4fayqq7rk2zyhuecwwbzvnliyasnxefkrl7erk.py
# Topologically Sorted Source Nodes: [sigmoid_13, mul_13, features_11_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_11_conv_6_avg_pool => mean
#   mul_13 => mul_81
#   sigmoid_13 => sigmoid_12
# Graph fragment:
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_81 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_12), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_81, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_17 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_17(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r2 + 4096*x1), xmask, other=0.0)
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


# kernel path: inductor_cache/wq/cwqk35n7pcdv2ohok4vxmlsato7pfgdz6pum2a3jhu6upax7n3k5.py
# Topologically Sorted Source Nodes: [sigmoid_14, mul_14], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_14 => mul_82
#   sigmoid_14 => sigmoid_13
# Graph fragment:
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm,), kwargs = {})
#   %mul_82 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, %sigmoid_13), kwargs = {})
triton_poi_fused_mul_sigmoid_18 = async_compile.triton('triton_poi_fused_mul_sigmoid_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/55/c5564bmu4tka7jzebxikzzmer6othansxi3wwqfyu4xfrlgvx4s2.py
# Topologically Sorted Source Nodes: [sigmoid_13, mul_13, mul_15], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_13 => mul_81
#   mul_15 => mul_83
#   sigmoid_13 => sigmoid_12
# Graph fragment:
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_53,), kwargs = {})
#   %mul_81 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_53, %sigmoid_12), kwargs = {})
#   %mul_83 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %view_1), kwargs = {})
triton_poi_fused_mul_sigmoid_19 = async_compile.triton('triton_poi_fused_mul_sigmoid_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_19(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 256)
    x2 = xindex // 4096
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 256*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/rx/crxg7bhf3olrmuxyyahfdzgnx55lirrrptmq45ldtvqalbiysqjy.py
# Topologically Sorted Source Nodes: [features_11_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_11_conv_8 => add_55, mul_85, mul_86, sub_23
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_189), kwargs = {})
#   %add_55 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_191), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_20(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
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


# kernel path: inductor_cache/ll/cllpcufhhireosukpdhejizwh2jp4mtjzop5ukzyrrxy4ux4uxmi.py
# Topologically Sorted Source Nodes: [features_12_conv_1, sigmoid_17, mul_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_12_conv_1 => add_57, mul_88, mul_89, sub_24
#   mul_16 => mul_90
#   sigmoid_17 => sigmoid_15
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_88 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %unsqueeze_197), kwargs = {})
#   %add_57 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %unsqueeze_199), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_57,), kwargs = {})
#   %mul_90 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_57, %sigmoid_15), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ez/cezlsmi5bqz7bkwr2r74l2vxi7qi4oushz3yuylugrj4xxnf3qzi.py
# Topologically Sorted Source Nodes: [features_12_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_12_conv_4 => add_59, mul_92, mul_93, sub_25
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_92, %unsqueeze_205), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %unsqueeze_207), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
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


# kernel path: inductor_cache/46/c4646rma6e5jgpih6ce6grq3zqsdgjkiupznch3q4pbqssvhlzyw.py
# Topologically Sorted Source Nodes: [sigmoid_19, mul_17, features_12_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_12_conv_6_avg_pool => mean_1
#   mul_17 => mul_94
#   sigmoid_19 => sigmoid_16
# Graph fragment:
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_59,), kwargs = {})
#   %mul_94 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, %sigmoid_16), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_94, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_23 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_23(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 512)
    x1 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 512*r2 + 8192*x1), xmask, other=0.0)
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


# kernel path: inductor_cache/ot/cotlf7mqx3a42slnj7jbgugdhc5iigpg7bentqxvfzhnhw7txecc.py
# Topologically Sorted Source Nodes: [sigmoid_21, mul_18], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_18 => mul_95
#   sigmoid_21 => sigmoid_17
# Graph fragment:
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_2,), kwargs = {})
#   %mul_95 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_2, %sigmoid_17), kwargs = {})
triton_poi_fused_mul_sigmoid_24 = async_compile.triton('triton_poi_fused_mul_sigmoid_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_24(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ix/cixf4n3wixppjsb5baqaqane2irukwgnzoigtm3fijfgpei2dtni.py
# Topologically Sorted Source Nodes: [sigmoid_19, mul_17, mul_19], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_17 => mul_94
#   mul_19 => mul_96
#   sigmoid_19 => sigmoid_16
# Graph fragment:
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_59,), kwargs = {})
#   %mul_94 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_59, %sigmoid_16), kwargs = {})
#   %mul_96 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_94, %view_3), kwargs = {})
triton_poi_fused_mul_sigmoid_25 = async_compile.triton('triton_poi_fused_mul_sigmoid_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_25(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 512)
    x2 = xindex // 8192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 512*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/6k/c6kdoc5siqt7xgetwyalxnqjbmbmcw7p6xas7kxru6pvngbm3x7f.py
# Topologically Sorted Source Nodes: [features_12_conv_8, add_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_9 => add_62
#   features_12_conv_8 => add_61, mul_98, mul_99, sub_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_98 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_99 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_98, %unsqueeze_213), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_99, %unsqueeze_215), kwargs = {})
#   %add_62 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_55, %add_61), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
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


# kernel path: inductor_cache/md/cmdh5ig2pztd3hlwvw6f7xvfkbyhdbmmbkeprprnd7stm3ct77va.py
# Topologically Sorted Source Nodes: [features_17_conv_1, sigmoid_52, mul_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_17_conv_1 => add_92, mul_153, mul_154, sub_39
#   mul_36 => mul_155
#   sigmoid_52 => sigmoid_35
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_153, %unsqueeze_317), kwargs = {})
#   %add_92 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_154, %unsqueeze_319), kwargs = {})
#   %sigmoid_35 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_92,), kwargs = {})
#   %mul_155 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %sigmoid_35), kwargs = {})
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
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
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


# kernel path: inductor_cache/sf/csf3assikwcwdjuwa56bidd4qvekprg53peln6elgbrul3ue6unn.py
# Topologically Sorted Source Nodes: [features_17_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_17_conv_4 => add_94, mul_157, mul_158, sub_40
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_321), kwargs = {})
#   %mul_157 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_157, %unsqueeze_325), kwargs = {})
#   %add_94 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_158, %unsqueeze_327), kwargs = {})
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
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
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


# kernel path: inductor_cache/nl/cnlzldeoieo2fxkgt25vkfmb2tv7s6bo5o3y6t24d4vf7hxgprcu.py
# Topologically Sorted Source Nodes: [sigmoid_54, mul_37, features_17_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_17_conv_6_avg_pool => mean_6
#   mul_37 => mul_159
#   sigmoid_54 => sigmoid_36
# Graph fragment:
#   %sigmoid_36 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_94,), kwargs = {})
#   %mul_159 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, %sigmoid_36), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_159, [-1, -2], True), kwargs = {})
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
    xnumel = 3072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 768)
    x1 = xindex // 768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*r2 + 12288*x1), xmask, other=0.0)
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


# kernel path: inductor_cache/7z/c7zy6ek4p53nrgr3wjnaaljga75s5peuccipfgl57xkdzvqrv6u5.py
# Topologically Sorted Source Nodes: [sigmoid_54, mul_37, mul_39], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_37 => mul_159
#   mul_39 => mul_161
#   sigmoid_54 => sigmoid_36
# Graph fragment:
#   %sigmoid_36 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_94,), kwargs = {})
#   %mul_159 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_94, %sigmoid_36), kwargs = {})
#   %mul_161 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_159, %view_13), kwargs = {})
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
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 768)
    x2 = xindex // 12288
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 768*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/gr/cgrxlmvwaniv6v467w5ctbhwufqqlzqnnx524gm54owst7nalxol.py
# Topologically Sorted Source Nodes: [features_17_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_17_conv_8 => add_96, mul_163, mul_164, sub_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_163, %unsqueeze_333), kwargs = {})
#   %add_96 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_164, %unsqueeze_335), kwargs = {})
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


# kernel path: inductor_cache/kn/cknaymyve7vh3l73ny5y6t2wnmfxvl56rau65ccuctvgq7zcvdb5.py
# Topologically Sorted Source Nodes: [features_18_conv_1, sigmoid_59, mul_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_18_conv_1 => add_98, mul_166, mul_167, sub_42
#   mul_40 => mul_168
#   sigmoid_59 => sigmoid_39
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_166, %unsqueeze_341), kwargs = {})
#   %add_98 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_167, %unsqueeze_343), kwargs = {})
#   %sigmoid_39 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_98,), kwargs = {})
#   %mul_168 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_98, %sigmoid_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/2h/c2hksnl6oeqese4ovdiivyntdhszbjxksplwcsijyq5b3vzg7cut.py
# Topologically Sorted Source Nodes: [features_18_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_18_conv_4 => add_100, mul_170, mul_171, sub_43
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_170 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_171 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_170, %unsqueeze_349), kwargs = {})
#   %add_100 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_171, %unsqueeze_351), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ne/cneql2e7oetnqgzfw6ofx4keihflldsomjogafocguwdacsottvv.py
# Topologically Sorted Source Nodes: [sigmoid_61, mul_41, features_18_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_18_conv_6_avg_pool => mean_7
#   mul_41 => mul_172
#   sigmoid_61 => sigmoid_40
# Graph fragment:
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_100,), kwargs = {})
#   %mul_172 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, %sigmoid_40), kwargs = {})
#   %mean_7 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_172, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_34 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_34(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/2x/c2xey4dqas7b4y3h7hkire5vvnyqqg63vguo5qxyahf73g7hbmdf.py
# Topologically Sorted Source Nodes: [sigmoid_63, mul_42], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_42 => mul_173
#   sigmoid_63 => sigmoid_41
# Graph fragment:
#   %sigmoid_41 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_14,), kwargs = {})
#   %mul_173 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_14, %sigmoid_41), kwargs = {})
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


# kernel path: inductor_cache/hs/chsm2quqthfjcmydkjvogw35eqywnij3xts43rmlrujes2ktcgsm.py
# Topologically Sorted Source Nodes: [sigmoid_61, mul_41, mul_43], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_41 => mul_172
#   mul_43 => mul_174
#   sigmoid_61 => sigmoid_40
# Graph fragment:
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_100,), kwargs = {})
#   %mul_172 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_100, %sigmoid_40), kwargs = {})
#   %mul_174 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_172, %view_15), kwargs = {})
triton_poi_fused_mul_sigmoid_36 = async_compile.triton('triton_poi_fused_mul_sigmoid_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/yh/cyhjrhlzdnvrnaf35sfzidip4l3bax4oex3sdtmc7hufgronplha.py
# Topologically Sorted Source Nodes: [features_18_conv_8, add_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_14 => add_103
#   features_18_conv_8 => add_102, mul_176, mul_177, sub_44
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_176, %unsqueeze_357), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_177, %unsqueeze_359), kwargs = {})
#   %add_103 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_96, %add_102), kwargs = {})
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


# kernel path: inductor_cache/4k/c4k2if3xl6lnocmrchy5zwzg7hqnv5wtowctssbi43ynmqlarc2y.py
# Topologically Sorted Source Nodes: [features_26_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_26_conv_4 => add_156, mul_274, mul_275, sub_67
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %unsqueeze_541), kwargs = {})
#   %add_156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %unsqueeze_543), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 960)
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


# kernel path: inductor_cache/nv/cnvegght5ebu7zixf6zn2ih2gbo56m3qxyfkbb7ujzzx5ubvqvvr.py
# Topologically Sorted Source Nodes: [sigmoid_117, mul_73, features_26_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_26_conv_6_avg_pool => mean_15
#   mul_73 => mul_276
#   sigmoid_117 => sigmoid_72
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_156,), kwargs = {})
#   %mul_276 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, %sigmoid_72), kwargs = {})
#   %mean_15 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_276, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_39 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_39', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_39(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 960)
    x1 = xindex // 960
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 3840*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (960 + x0 + 3840*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (1920 + x0 + 3840*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (2880 + x0 + 3840*x1), xmask)
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


# kernel path: inductor_cache/cd/ccdxyatcwcxvjlypusm6h63pv6aepjwwirfvo2mymxjwlirlyepj.py
# Topologically Sorted Source Nodes: [sigmoid_117, mul_73, mul_75], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_73 => mul_276
#   mul_75 => mul_278
#   sigmoid_117 => sigmoid_72
# Graph fragment:
#   %sigmoid_72 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_156,), kwargs = {})
#   %mul_276 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_156, %sigmoid_72), kwargs = {})
#   %mul_278 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_276, %view_31), kwargs = {})
triton_poi_fused_mul_sigmoid_40 = async_compile.triton('triton_poi_fused_mul_sigmoid_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_40(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 960)
    x2 = xindex // 3840
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 960*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s2/cs2m3rrfzregauyefxwhgigehho2rajeobzr4n4ncuv4ydwbqp4w.py
# Topologically Sorted Source Nodes: [features_26_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_26_conv_8 => add_158, mul_280, mul_281, sub_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_280 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %unsqueeze_549), kwargs = {})
#   %add_158 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_281, %unsqueeze_551), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_41', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
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


# kernel path: inductor_cache/rs/crsop5s3o5q2e4u4zhehhcsovn4ebz7ls5sdtzc2plm4yhjwmtrb.py
# Topologically Sorted Source Nodes: [features_27_conv_1, sigmoid_122, mul_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_27_conv_1 => add_160, mul_283, mul_284, sub_69
#   mul_76 => mul_285
#   sigmoid_122 => sigmoid_75
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_553), kwargs = {})
#   %mul_283 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_283, %unsqueeze_557), kwargs = {})
#   %add_160 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_284, %unsqueeze_559), kwargs = {})
#   %sigmoid_75 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_160,), kwargs = {})
#   %mul_285 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_160, %sigmoid_75), kwargs = {})
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
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1536)
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


# kernel path: inductor_cache/3c/c3ctt3j4s6yvxtyllnadczfngzqtzztvi667dp2zotmvrorv2ad2.py
# Topologically Sorted Source Nodes: [features_27_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_27_conv_4 => add_162, mul_287, mul_288, sub_70
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_561), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_287, %unsqueeze_565), kwargs = {})
#   %add_162 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_288, %unsqueeze_567), kwargs = {})
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
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1536)
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


# kernel path: inductor_cache/sz/csz22sskz4rnzazjgycn2calkpgb7mxoux7fgbgvvjr63xuuhust.py
# Topologically Sorted Source Nodes: [sigmoid_124, mul_77, features_27_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_27_conv_6_avg_pool => mean_16
#   mul_77 => mul_289
#   sigmoid_124 => sigmoid_76
# Graph fragment:
#   %sigmoid_76 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_162,), kwargs = {})
#   %mul_289 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_162, %sigmoid_76), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_289, [-1, -2], True), kwargs = {})
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
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 6144*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (1536 + x0 + 6144*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (3072 + x0 + 6144*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (4608 + x0 + 6144*x1), xmask)
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


# kernel path: inductor_cache/p7/cp72h3onxnkxzdgv5xcm5cnnv3vjmc6wfend55xw3iggfcu6bsg6.py
# Topologically Sorted Source Nodes: [sigmoid_126, mul_78], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_78 => mul_290
#   sigmoid_126 => sigmoid_77
# Graph fragment:
#   %sigmoid_77 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_32,), kwargs = {})
#   %mul_290 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_32, %sigmoid_77), kwargs = {})
triton_poi_fused_mul_sigmoid_45 = async_compile.triton('triton_poi_fused_mul_sigmoid_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_45', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_45(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rd/crdml32jxzgaqlig7pz52ynrc4pwufqvmoxwfojptvoxxjqqnxhz.py
# Topologically Sorted Source Nodes: [sigmoid_124, mul_77, mul_79], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_77 => mul_289
#   mul_79 => mul_291
#   sigmoid_124 => sigmoid_76
# Graph fragment:
#   %sigmoid_76 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_162,), kwargs = {})
#   %mul_289 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_162, %sigmoid_76), kwargs = {})
#   %mul_291 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_289, %view_33), kwargs = {})
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
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1536)
    x2 = xindex // 6144
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 1536*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/d4/cd44n363g6ual7eap3rcz672sbgmi6zxkovtm6cx3ob36flhjnla.py
# Topologically Sorted Source Nodes: [features_27_conv_8, add_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_22 => add_165
#   features_27_conv_8 => add_164, mul_293, mul_294, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_569), kwargs = {})
#   %mul_293 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_294 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_293, %unsqueeze_573), kwargs = {})
#   %add_164 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_294, %unsqueeze_575), kwargs = {})
#   %add_165 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_158, %add_164), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_47', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_47(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
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


# kernel path: inductor_cache/ho/chojrgzjq433och7ipsbm6tmnhxdr7y2jun5t63ph3wbatulalzw.py
# Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv_1 => add_258, mul_465, mul_466, sub_111
# Graph fragment:
#   %sub_111 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_111, %unsqueeze_889), kwargs = {})
#   %mul_465 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_111, %unsqueeze_891), kwargs = {})
#   %mul_466 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_465, %unsqueeze_893), kwargs = {})
#   %add_258 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_466, %unsqueeze_895), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wt/cwtdawumkezbwtyuffa6crom6fdwwkzrgb63wy5erk6bmn2xeoh7.py
# Topologically Sorted Source Nodes: [sigmoid_220, mul_132, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   avgpool => mean_30
#   mul_132 => mul_467
#   sigmoid_220 => sigmoid_131
# Graph fragment:
#   %sigmoid_131 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_258,), kwargs = {})
#   %mul_467 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_258, %sigmoid_131), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_467, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_49 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_49', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683 = args
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
    assert_size_stride(primals_27, (96, 24, 3, 3), (216, 9, 3, 1))
    assert_size_stride(primals_28, (96, ), (1, ))
    assert_size_stride(primals_29, (96, ), (1, ))
    assert_size_stride(primals_30, (96, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_32, (48, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_33, (48, ), (1, ))
    assert_size_stride(primals_34, (48, ), (1, ))
    assert_size_stride(primals_35, (48, ), (1, ))
    assert_size_stride(primals_36, (48, ), (1, ))
    assert_size_stride(primals_37, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (192, ), (1, ))
    assert_size_stride(primals_42, (48, 192, 1, 1), (192, 1, 1, 1))
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
    assert_size_stride(primals_72, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_73, (64, ), (1, ))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, ), (1, ))
    assert_size_stride(primals_77, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (64, ), (1, ))
    assert_size_stride(primals_87, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (256, ), (1, ))
    assert_size_stride(primals_91, (256, ), (1, ))
    assert_size_stride(primals_92, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, ), (1, ))
    assert_size_stride(primals_97, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_98, (256, ), (1, ))
    assert_size_stride(primals_99, (256, ), (1, ))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_103, (64, ), (1, ))
    assert_size_stride(primals_104, (64, ), (1, ))
    assert_size_stride(primals_105, (64, ), (1, ))
    assert_size_stride(primals_106, (64, ), (1, ))
    assert_size_stride(primals_107, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, ), (1, ))
    assert_size_stride(primals_116, (256, ), (1, ))
    assert_size_stride(primals_117, (16, 256), (256, 1))
    assert_size_stride(primals_118, (16, ), (1, ))
    assert_size_stride(primals_119, (256, 16), (16, 1))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (512, ), (1, ))
    assert_size_stride(primals_129, (512, ), (1, ))
    assert_size_stride(primals_130, (512, ), (1, ))
    assert_size_stride(primals_131, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (512, ), (1, ))
    assert_size_stride(primals_136, (32, 512), (512, 1))
    assert_size_stride(primals_137, (32, ), (1, ))
    assert_size_stride(primals_138, (512, 32), (32, 1))
    assert_size_stride(primals_139, (512, ), (1, ))
    assert_size_stride(primals_140, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, ), (1, ))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_146, (512, ), (1, ))
    assert_size_stride(primals_147, (512, ), (1, ))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_151, (512, ), (1, ))
    assert_size_stride(primals_152, (512, ), (1, ))
    assert_size_stride(primals_153, (512, ), (1, ))
    assert_size_stride(primals_154, (512, ), (1, ))
    assert_size_stride(primals_155, (32, 512), (512, 1))
    assert_size_stride(primals_156, (32, ), (1, ))
    assert_size_stride(primals_157, (512, 32), (32, 1))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_159, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_160, (128, ), (1, ))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_168, (512, ), (1, ))
    assert_size_stride(primals_169, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (512, ), (1, ))
    assert_size_stride(primals_171, (512, ), (1, ))
    assert_size_stride(primals_172, (512, ), (1, ))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_174, (32, 512), (512, 1))
    assert_size_stride(primals_175, (32, ), (1, ))
    assert_size_stride(primals_176, (512, 32), (32, 1))
    assert_size_stride(primals_177, (512, ), (1, ))
    assert_size_stride(primals_178, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_179, (128, ), (1, ))
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_184, (512, ), (1, ))
    assert_size_stride(primals_185, (512, ), (1, ))
    assert_size_stride(primals_186, (512, ), (1, ))
    assert_size_stride(primals_187, (512, ), (1, ))
    assert_size_stride(primals_188, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_189, (512, ), (1, ))
    assert_size_stride(primals_190, (512, ), (1, ))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (512, ), (1, ))
    assert_size_stride(primals_193, (32, 512), (512, 1))
    assert_size_stride(primals_194, (32, ), (1, ))
    assert_size_stride(primals_195, (512, 32), (32, 1))
    assert_size_stride(primals_196, (512, ), (1, ))
    assert_size_stride(primals_197, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_198, (128, ), (1, ))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_201, (128, ), (1, ))
    assert_size_stride(primals_202, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (512, ), (1, ))
    assert_size_stride(primals_207, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_208, (512, ), (1, ))
    assert_size_stride(primals_209, (512, ), (1, ))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (512, ), (1, ))
    assert_size_stride(primals_212, (32, 512), (512, 1))
    assert_size_stride(primals_213, (32, ), (1, ))
    assert_size_stride(primals_214, (512, 32), (32, 1))
    assert_size_stride(primals_215, (512, ), (1, ))
    assert_size_stride(primals_216, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (128, ), (1, ))
    assert_size_stride(primals_220, (128, ), (1, ))
    assert_size_stride(primals_221, (768, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_222, (768, ), (1, ))
    assert_size_stride(primals_223, (768, ), (1, ))
    assert_size_stride(primals_224, (768, ), (1, ))
    assert_size_stride(primals_225, (768, ), (1, ))
    assert_size_stride(primals_226, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (768, ), (1, ))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_231, (32, 768), (768, 1))
    assert_size_stride(primals_232, (32, ), (1, ))
    assert_size_stride(primals_233, (768, 32), (32, 1))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_236, (160, ), (1, ))
    assert_size_stride(primals_237, (160, ), (1, ))
    assert_size_stride(primals_238, (160, ), (1, ))
    assert_size_stride(primals_239, (160, ), (1, ))
    assert_size_stride(primals_240, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_241, (960, ), (1, ))
    assert_size_stride(primals_242, (960, ), (1, ))
    assert_size_stride(primals_243, (960, ), (1, ))
    assert_size_stride(primals_244, (960, ), (1, ))
    assert_size_stride(primals_245, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_246, (960, ), (1, ))
    assert_size_stride(primals_247, (960, ), (1, ))
    assert_size_stride(primals_248, (960, ), (1, ))
    assert_size_stride(primals_249, (960, ), (1, ))
    assert_size_stride(primals_250, (40, 960), (960, 1))
    assert_size_stride(primals_251, (40, ), (1, ))
    assert_size_stride(primals_252, (960, 40), (40, 1))
    assert_size_stride(primals_253, (960, ), (1, ))
    assert_size_stride(primals_254, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_255, (160, ), (1, ))
    assert_size_stride(primals_256, (160, ), (1, ))
    assert_size_stride(primals_257, (160, ), (1, ))
    assert_size_stride(primals_258, (160, ), (1, ))
    assert_size_stride(primals_259, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_260, (960, ), (1, ))
    assert_size_stride(primals_261, (960, ), (1, ))
    assert_size_stride(primals_262, (960, ), (1, ))
    assert_size_stride(primals_263, (960, ), (1, ))
    assert_size_stride(primals_264, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_265, (960, ), (1, ))
    assert_size_stride(primals_266, (960, ), (1, ))
    assert_size_stride(primals_267, (960, ), (1, ))
    assert_size_stride(primals_268, (960, ), (1, ))
    assert_size_stride(primals_269, (40, 960), (960, 1))
    assert_size_stride(primals_270, (40, ), (1, ))
    assert_size_stride(primals_271, (960, 40), (40, 1))
    assert_size_stride(primals_272, (960, ), (1, ))
    assert_size_stride(primals_273, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_274, (160, ), (1, ))
    assert_size_stride(primals_275, (160, ), (1, ))
    assert_size_stride(primals_276, (160, ), (1, ))
    assert_size_stride(primals_277, (160, ), (1, ))
    assert_size_stride(primals_278, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_279, (960, ), (1, ))
    assert_size_stride(primals_280, (960, ), (1, ))
    assert_size_stride(primals_281, (960, ), (1, ))
    assert_size_stride(primals_282, (960, ), (1, ))
    assert_size_stride(primals_283, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_284, (960, ), (1, ))
    assert_size_stride(primals_285, (960, ), (1, ))
    assert_size_stride(primals_286, (960, ), (1, ))
    assert_size_stride(primals_287, (960, ), (1, ))
    assert_size_stride(primals_288, (40, 960), (960, 1))
    assert_size_stride(primals_289, (40, ), (1, ))
    assert_size_stride(primals_290, (960, 40), (40, 1))
    assert_size_stride(primals_291, (960, ), (1, ))
    assert_size_stride(primals_292, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_293, (160, ), (1, ))
    assert_size_stride(primals_294, (160, ), (1, ))
    assert_size_stride(primals_295, (160, ), (1, ))
    assert_size_stride(primals_296, (160, ), (1, ))
    assert_size_stride(primals_297, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_298, (960, ), (1, ))
    assert_size_stride(primals_299, (960, ), (1, ))
    assert_size_stride(primals_300, (960, ), (1, ))
    assert_size_stride(primals_301, (960, ), (1, ))
    assert_size_stride(primals_302, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_303, (960, ), (1, ))
    assert_size_stride(primals_304, (960, ), (1, ))
    assert_size_stride(primals_305, (960, ), (1, ))
    assert_size_stride(primals_306, (960, ), (1, ))
    assert_size_stride(primals_307, (40, 960), (960, 1))
    assert_size_stride(primals_308, (40, ), (1, ))
    assert_size_stride(primals_309, (960, 40), (40, 1))
    assert_size_stride(primals_310, (960, ), (1, ))
    assert_size_stride(primals_311, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_312, (160, ), (1, ))
    assert_size_stride(primals_313, (160, ), (1, ))
    assert_size_stride(primals_314, (160, ), (1, ))
    assert_size_stride(primals_315, (160, ), (1, ))
    assert_size_stride(primals_316, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_317, (960, ), (1, ))
    assert_size_stride(primals_318, (960, ), (1, ))
    assert_size_stride(primals_319, (960, ), (1, ))
    assert_size_stride(primals_320, (960, ), (1, ))
    assert_size_stride(primals_321, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_322, (960, ), (1, ))
    assert_size_stride(primals_323, (960, ), (1, ))
    assert_size_stride(primals_324, (960, ), (1, ))
    assert_size_stride(primals_325, (960, ), (1, ))
    assert_size_stride(primals_326, (40, 960), (960, 1))
    assert_size_stride(primals_327, (40, ), (1, ))
    assert_size_stride(primals_328, (960, 40), (40, 1))
    assert_size_stride(primals_329, (960, ), (1, ))
    assert_size_stride(primals_330, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_331, (160, ), (1, ))
    assert_size_stride(primals_332, (160, ), (1, ))
    assert_size_stride(primals_333, (160, ), (1, ))
    assert_size_stride(primals_334, (160, ), (1, ))
    assert_size_stride(primals_335, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_336, (960, ), (1, ))
    assert_size_stride(primals_337, (960, ), (1, ))
    assert_size_stride(primals_338, (960, ), (1, ))
    assert_size_stride(primals_339, (960, ), (1, ))
    assert_size_stride(primals_340, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_341, (960, ), (1, ))
    assert_size_stride(primals_342, (960, ), (1, ))
    assert_size_stride(primals_343, (960, ), (1, ))
    assert_size_stride(primals_344, (960, ), (1, ))
    assert_size_stride(primals_345, (40, 960), (960, 1))
    assert_size_stride(primals_346, (40, ), (1, ))
    assert_size_stride(primals_347, (960, 40), (40, 1))
    assert_size_stride(primals_348, (960, ), (1, ))
    assert_size_stride(primals_349, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_350, (160, ), (1, ))
    assert_size_stride(primals_351, (160, ), (1, ))
    assert_size_stride(primals_352, (160, ), (1, ))
    assert_size_stride(primals_353, (160, ), (1, ))
    assert_size_stride(primals_354, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_355, (960, ), (1, ))
    assert_size_stride(primals_356, (960, ), (1, ))
    assert_size_stride(primals_357, (960, ), (1, ))
    assert_size_stride(primals_358, (960, ), (1, ))
    assert_size_stride(primals_359, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_360, (960, ), (1, ))
    assert_size_stride(primals_361, (960, ), (1, ))
    assert_size_stride(primals_362, (960, ), (1, ))
    assert_size_stride(primals_363, (960, ), (1, ))
    assert_size_stride(primals_364, (40, 960), (960, 1))
    assert_size_stride(primals_365, (40, ), (1, ))
    assert_size_stride(primals_366, (960, 40), (40, 1))
    assert_size_stride(primals_367, (960, ), (1, ))
    assert_size_stride(primals_368, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_369, (160, ), (1, ))
    assert_size_stride(primals_370, (160, ), (1, ))
    assert_size_stride(primals_371, (160, ), (1, ))
    assert_size_stride(primals_372, (160, ), (1, ))
    assert_size_stride(primals_373, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_374, (960, ), (1, ))
    assert_size_stride(primals_375, (960, ), (1, ))
    assert_size_stride(primals_376, (960, ), (1, ))
    assert_size_stride(primals_377, (960, ), (1, ))
    assert_size_stride(primals_378, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_379, (960, ), (1, ))
    assert_size_stride(primals_380, (960, ), (1, ))
    assert_size_stride(primals_381, (960, ), (1, ))
    assert_size_stride(primals_382, (960, ), (1, ))
    assert_size_stride(primals_383, (40, 960), (960, 1))
    assert_size_stride(primals_384, (40, ), (1, ))
    assert_size_stride(primals_385, (960, 40), (40, 1))
    assert_size_stride(primals_386, (960, ), (1, ))
    assert_size_stride(primals_387, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_388, (160, ), (1, ))
    assert_size_stride(primals_389, (160, ), (1, ))
    assert_size_stride(primals_390, (160, ), (1, ))
    assert_size_stride(primals_391, (160, ), (1, ))
    assert_size_stride(primals_392, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_393, (960, ), (1, ))
    assert_size_stride(primals_394, (960, ), (1, ))
    assert_size_stride(primals_395, (960, ), (1, ))
    assert_size_stride(primals_396, (960, ), (1, ))
    assert_size_stride(primals_397, (960, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_398, (960, ), (1, ))
    assert_size_stride(primals_399, (960, ), (1, ))
    assert_size_stride(primals_400, (960, ), (1, ))
    assert_size_stride(primals_401, (960, ), (1, ))
    assert_size_stride(primals_402, (40, 960), (960, 1))
    assert_size_stride(primals_403, (40, ), (1, ))
    assert_size_stride(primals_404, (960, 40), (40, 1))
    assert_size_stride(primals_405, (960, ), (1, ))
    assert_size_stride(primals_406, (256, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_407, (256, ), (1, ))
    assert_size_stride(primals_408, (256, ), (1, ))
    assert_size_stride(primals_409, (256, ), (1, ))
    assert_size_stride(primals_410, (256, ), (1, ))
    assert_size_stride(primals_411, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_412, (1536, ), (1, ))
    assert_size_stride(primals_413, (1536, ), (1, ))
    assert_size_stride(primals_414, (1536, ), (1, ))
    assert_size_stride(primals_415, (1536, ), (1, ))
    assert_size_stride(primals_416, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_417, (1536, ), (1, ))
    assert_size_stride(primals_418, (1536, ), (1, ))
    assert_size_stride(primals_419, (1536, ), (1, ))
    assert_size_stride(primals_420, (1536, ), (1, ))
    assert_size_stride(primals_421, (64, 1536), (1536, 1))
    assert_size_stride(primals_422, (64, ), (1, ))
    assert_size_stride(primals_423, (1536, 64), (64, 1))
    assert_size_stride(primals_424, (1536, ), (1, ))
    assert_size_stride(primals_425, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_426, (256, ), (1, ))
    assert_size_stride(primals_427, (256, ), (1, ))
    assert_size_stride(primals_428, (256, ), (1, ))
    assert_size_stride(primals_429, (256, ), (1, ))
    assert_size_stride(primals_430, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_431, (1536, ), (1, ))
    assert_size_stride(primals_432, (1536, ), (1, ))
    assert_size_stride(primals_433, (1536, ), (1, ))
    assert_size_stride(primals_434, (1536, ), (1, ))
    assert_size_stride(primals_435, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_436, (1536, ), (1, ))
    assert_size_stride(primals_437, (1536, ), (1, ))
    assert_size_stride(primals_438, (1536, ), (1, ))
    assert_size_stride(primals_439, (1536, ), (1, ))
    assert_size_stride(primals_440, (64, 1536), (1536, 1))
    assert_size_stride(primals_441, (64, ), (1, ))
    assert_size_stride(primals_442, (1536, 64), (64, 1))
    assert_size_stride(primals_443, (1536, ), (1, ))
    assert_size_stride(primals_444, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_445, (256, ), (1, ))
    assert_size_stride(primals_446, (256, ), (1, ))
    assert_size_stride(primals_447, (256, ), (1, ))
    assert_size_stride(primals_448, (256, ), (1, ))
    assert_size_stride(primals_449, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_450, (1536, ), (1, ))
    assert_size_stride(primals_451, (1536, ), (1, ))
    assert_size_stride(primals_452, (1536, ), (1, ))
    assert_size_stride(primals_453, (1536, ), (1, ))
    assert_size_stride(primals_454, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_455, (1536, ), (1, ))
    assert_size_stride(primals_456, (1536, ), (1, ))
    assert_size_stride(primals_457, (1536, ), (1, ))
    assert_size_stride(primals_458, (1536, ), (1, ))
    assert_size_stride(primals_459, (64, 1536), (1536, 1))
    assert_size_stride(primals_460, (64, ), (1, ))
    assert_size_stride(primals_461, (1536, 64), (64, 1))
    assert_size_stride(primals_462, (1536, ), (1, ))
    assert_size_stride(primals_463, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_464, (256, ), (1, ))
    assert_size_stride(primals_465, (256, ), (1, ))
    assert_size_stride(primals_466, (256, ), (1, ))
    assert_size_stride(primals_467, (256, ), (1, ))
    assert_size_stride(primals_468, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_469, (1536, ), (1, ))
    assert_size_stride(primals_470, (1536, ), (1, ))
    assert_size_stride(primals_471, (1536, ), (1, ))
    assert_size_stride(primals_472, (1536, ), (1, ))
    assert_size_stride(primals_473, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_474, (1536, ), (1, ))
    assert_size_stride(primals_475, (1536, ), (1, ))
    assert_size_stride(primals_476, (1536, ), (1, ))
    assert_size_stride(primals_477, (1536, ), (1, ))
    assert_size_stride(primals_478, (64, 1536), (1536, 1))
    assert_size_stride(primals_479, (64, ), (1, ))
    assert_size_stride(primals_480, (1536, 64), (64, 1))
    assert_size_stride(primals_481, (1536, ), (1, ))
    assert_size_stride(primals_482, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_483, (256, ), (1, ))
    assert_size_stride(primals_484, (256, ), (1, ))
    assert_size_stride(primals_485, (256, ), (1, ))
    assert_size_stride(primals_486, (256, ), (1, ))
    assert_size_stride(primals_487, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_488, (1536, ), (1, ))
    assert_size_stride(primals_489, (1536, ), (1, ))
    assert_size_stride(primals_490, (1536, ), (1, ))
    assert_size_stride(primals_491, (1536, ), (1, ))
    assert_size_stride(primals_492, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_493, (1536, ), (1, ))
    assert_size_stride(primals_494, (1536, ), (1, ))
    assert_size_stride(primals_495, (1536, ), (1, ))
    assert_size_stride(primals_496, (1536, ), (1, ))
    assert_size_stride(primals_497, (64, 1536), (1536, 1))
    assert_size_stride(primals_498, (64, ), (1, ))
    assert_size_stride(primals_499, (1536, 64), (64, 1))
    assert_size_stride(primals_500, (1536, ), (1, ))
    assert_size_stride(primals_501, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_502, (256, ), (1, ))
    assert_size_stride(primals_503, (256, ), (1, ))
    assert_size_stride(primals_504, (256, ), (1, ))
    assert_size_stride(primals_505, (256, ), (1, ))
    assert_size_stride(primals_506, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_507, (1536, ), (1, ))
    assert_size_stride(primals_508, (1536, ), (1, ))
    assert_size_stride(primals_509, (1536, ), (1, ))
    assert_size_stride(primals_510, (1536, ), (1, ))
    assert_size_stride(primals_511, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_512, (1536, ), (1, ))
    assert_size_stride(primals_513, (1536, ), (1, ))
    assert_size_stride(primals_514, (1536, ), (1, ))
    assert_size_stride(primals_515, (1536, ), (1, ))
    assert_size_stride(primals_516, (64, 1536), (1536, 1))
    assert_size_stride(primals_517, (64, ), (1, ))
    assert_size_stride(primals_518, (1536, 64), (64, 1))
    assert_size_stride(primals_519, (1536, ), (1, ))
    assert_size_stride(primals_520, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_521, (256, ), (1, ))
    assert_size_stride(primals_522, (256, ), (1, ))
    assert_size_stride(primals_523, (256, ), (1, ))
    assert_size_stride(primals_524, (256, ), (1, ))
    assert_size_stride(primals_525, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_526, (1536, ), (1, ))
    assert_size_stride(primals_527, (1536, ), (1, ))
    assert_size_stride(primals_528, (1536, ), (1, ))
    assert_size_stride(primals_529, (1536, ), (1, ))
    assert_size_stride(primals_530, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_531, (1536, ), (1, ))
    assert_size_stride(primals_532, (1536, ), (1, ))
    assert_size_stride(primals_533, (1536, ), (1, ))
    assert_size_stride(primals_534, (1536, ), (1, ))
    assert_size_stride(primals_535, (64, 1536), (1536, 1))
    assert_size_stride(primals_536, (64, ), (1, ))
    assert_size_stride(primals_537, (1536, 64), (64, 1))
    assert_size_stride(primals_538, (1536, ), (1, ))
    assert_size_stride(primals_539, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_540, (256, ), (1, ))
    assert_size_stride(primals_541, (256, ), (1, ))
    assert_size_stride(primals_542, (256, ), (1, ))
    assert_size_stride(primals_543, (256, ), (1, ))
    assert_size_stride(primals_544, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_545, (1536, ), (1, ))
    assert_size_stride(primals_546, (1536, ), (1, ))
    assert_size_stride(primals_547, (1536, ), (1, ))
    assert_size_stride(primals_548, (1536, ), (1, ))
    assert_size_stride(primals_549, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_550, (1536, ), (1, ))
    assert_size_stride(primals_551, (1536, ), (1, ))
    assert_size_stride(primals_552, (1536, ), (1, ))
    assert_size_stride(primals_553, (1536, ), (1, ))
    assert_size_stride(primals_554, (64, 1536), (1536, 1))
    assert_size_stride(primals_555, (64, ), (1, ))
    assert_size_stride(primals_556, (1536, 64), (64, 1))
    assert_size_stride(primals_557, (1536, ), (1, ))
    assert_size_stride(primals_558, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_559, (256, ), (1, ))
    assert_size_stride(primals_560, (256, ), (1, ))
    assert_size_stride(primals_561, (256, ), (1, ))
    assert_size_stride(primals_562, (256, ), (1, ))
    assert_size_stride(primals_563, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_564, (1536, ), (1, ))
    assert_size_stride(primals_565, (1536, ), (1, ))
    assert_size_stride(primals_566, (1536, ), (1, ))
    assert_size_stride(primals_567, (1536, ), (1, ))
    assert_size_stride(primals_568, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_569, (1536, ), (1, ))
    assert_size_stride(primals_570, (1536, ), (1, ))
    assert_size_stride(primals_571, (1536, ), (1, ))
    assert_size_stride(primals_572, (1536, ), (1, ))
    assert_size_stride(primals_573, (64, 1536), (1536, 1))
    assert_size_stride(primals_574, (64, ), (1, ))
    assert_size_stride(primals_575, (1536, 64), (64, 1))
    assert_size_stride(primals_576, (1536, ), (1, ))
    assert_size_stride(primals_577, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_578, (256, ), (1, ))
    assert_size_stride(primals_579, (256, ), (1, ))
    assert_size_stride(primals_580, (256, ), (1, ))
    assert_size_stride(primals_581, (256, ), (1, ))
    assert_size_stride(primals_582, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_583, (1536, ), (1, ))
    assert_size_stride(primals_584, (1536, ), (1, ))
    assert_size_stride(primals_585, (1536, ), (1, ))
    assert_size_stride(primals_586, (1536, ), (1, ))
    assert_size_stride(primals_587, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_588, (1536, ), (1, ))
    assert_size_stride(primals_589, (1536, ), (1, ))
    assert_size_stride(primals_590, (1536, ), (1, ))
    assert_size_stride(primals_591, (1536, ), (1, ))
    assert_size_stride(primals_592, (64, 1536), (1536, 1))
    assert_size_stride(primals_593, (64, ), (1, ))
    assert_size_stride(primals_594, (1536, 64), (64, 1))
    assert_size_stride(primals_595, (1536, ), (1, ))
    assert_size_stride(primals_596, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_597, (256, ), (1, ))
    assert_size_stride(primals_598, (256, ), (1, ))
    assert_size_stride(primals_599, (256, ), (1, ))
    assert_size_stride(primals_600, (256, ), (1, ))
    assert_size_stride(primals_601, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_602, (1536, ), (1, ))
    assert_size_stride(primals_603, (1536, ), (1, ))
    assert_size_stride(primals_604, (1536, ), (1, ))
    assert_size_stride(primals_605, (1536, ), (1, ))
    assert_size_stride(primals_606, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_607, (1536, ), (1, ))
    assert_size_stride(primals_608, (1536, ), (1, ))
    assert_size_stride(primals_609, (1536, ), (1, ))
    assert_size_stride(primals_610, (1536, ), (1, ))
    assert_size_stride(primals_611, (64, 1536), (1536, 1))
    assert_size_stride(primals_612, (64, ), (1, ))
    assert_size_stride(primals_613, (1536, 64), (64, 1))
    assert_size_stride(primals_614, (1536, ), (1, ))
    assert_size_stride(primals_615, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_616, (256, ), (1, ))
    assert_size_stride(primals_617, (256, ), (1, ))
    assert_size_stride(primals_618, (256, ), (1, ))
    assert_size_stride(primals_619, (256, ), (1, ))
    assert_size_stride(primals_620, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_621, (1536, ), (1, ))
    assert_size_stride(primals_622, (1536, ), (1, ))
    assert_size_stride(primals_623, (1536, ), (1, ))
    assert_size_stride(primals_624, (1536, ), (1, ))
    assert_size_stride(primals_625, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_626, (1536, ), (1, ))
    assert_size_stride(primals_627, (1536, ), (1, ))
    assert_size_stride(primals_628, (1536, ), (1, ))
    assert_size_stride(primals_629, (1536, ), (1, ))
    assert_size_stride(primals_630, (64, 1536), (1536, 1))
    assert_size_stride(primals_631, (64, ), (1, ))
    assert_size_stride(primals_632, (1536, 64), (64, 1))
    assert_size_stride(primals_633, (1536, ), (1, ))
    assert_size_stride(primals_634, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_635, (256, ), (1, ))
    assert_size_stride(primals_636, (256, ), (1, ))
    assert_size_stride(primals_637, (256, ), (1, ))
    assert_size_stride(primals_638, (256, ), (1, ))
    assert_size_stride(primals_639, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_640, (1536, ), (1, ))
    assert_size_stride(primals_641, (1536, ), (1, ))
    assert_size_stride(primals_642, (1536, ), (1, ))
    assert_size_stride(primals_643, (1536, ), (1, ))
    assert_size_stride(primals_644, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_645, (1536, ), (1, ))
    assert_size_stride(primals_646, (1536, ), (1, ))
    assert_size_stride(primals_647, (1536, ), (1, ))
    assert_size_stride(primals_648, (1536, ), (1, ))
    assert_size_stride(primals_649, (64, 1536), (1536, 1))
    assert_size_stride(primals_650, (64, ), (1, ))
    assert_size_stride(primals_651, (1536, 64), (64, 1))
    assert_size_stride(primals_652, (1536, ), (1, ))
    assert_size_stride(primals_653, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_654, (256, ), (1, ))
    assert_size_stride(primals_655, (256, ), (1, ))
    assert_size_stride(primals_656, (256, ), (1, ))
    assert_size_stride(primals_657, (256, ), (1, ))
    assert_size_stride(primals_658, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_659, (1536, ), (1, ))
    assert_size_stride(primals_660, (1536, ), (1, ))
    assert_size_stride(primals_661, (1536, ), (1, ))
    assert_size_stride(primals_662, (1536, ), (1, ))
    assert_size_stride(primals_663, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_664, (1536, ), (1, ))
    assert_size_stride(primals_665, (1536, ), (1, ))
    assert_size_stride(primals_666, (1536, ), (1, ))
    assert_size_stride(primals_667, (1536, ), (1, ))
    assert_size_stride(primals_668, (64, 1536), (1536, 1))
    assert_size_stride(primals_669, (64, ), (1, ))
    assert_size_stride(primals_670, (1536, 64), (64, 1))
    assert_size_stride(primals_671, (1536, ), (1, ))
    assert_size_stride(primals_672, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_673, (256, ), (1, ))
    assert_size_stride(primals_674, (256, ), (1, ))
    assert_size_stride(primals_675, (256, ), (1, ))
    assert_size_stride(primals_676, (256, ), (1, ))
    assert_size_stride(primals_677, (1792, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_678, (1792, ), (1, ))
    assert_size_stride(primals_679, (1792, ), (1, ))
    assert_size_stride(primals_680, (1792, ), (1, ))
    assert_size_stride(primals_681, (1792, ), (1, ))
    assert_size_stride(primals_682, (1000, 1792), (1792, 1))
    assert_size_stride(primals_683, (1000, ), (1, ))
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
        buf4 = empty_strided_cuda((96, 24, 3, 3), (216, 1, 72, 24), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_27, buf4, 2304, 9, grid=grid(2304, 9), stream=stream0)
        del primals_27
        buf5 = empty_strided_cuda((192, 48, 3, 3), (432, 1, 144, 48), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_37, buf5, 9216, 9, grid=grid(9216, 9), stream=stream0)
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
        buf9 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_77, buf9, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_77
        buf10 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_87, buf10, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_87
        buf11 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_97, buf11, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_97
        # Topologically Sorted Source Nodes: [features_0_0], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf13 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [features_0_1, sigmoid_1, mul_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf14, buf12, primals_3, primals_4, primals_5, primals_6, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_0], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf16 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf17 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [features_1_conv_1, sigmoid_2, mul_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf17, buf15, primals_8, primals_9, primals_10, primals_11, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_3], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf19 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        # Topologically Sorted Source Nodes: [features_1_conv_4, add_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf14, buf18, primals_13, primals_14, primals_15, primals_16, buf19, 98304, grid=grid(98304), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [features_2_conv_0], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf21 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [features_2_conv_1, sigmoid_3, mul_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_6.run(buf22, buf20, primals_18, primals_19, primals_20, primals_21, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_2_conv_3], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf24 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        # Topologically Sorted Source Nodes: [features_2_conv_4, add_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_7.run(buf19, buf23, primals_23, primals_24, primals_25, primals_26, buf24, 98304, grid=grid(98304), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [features_3_conv_0], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 96, 16, 16), (24576, 1, 1536, 96))
        buf26 = empty_strided_cuda((4, 96, 16, 16), (24576, 1, 1536, 96), torch.float32)
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [features_3_conv_1, sigmoid_4, mul_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_8.run(buf27, buf25, primals_28, primals_29, primals_30, primals_31, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_3_conv_3], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf29 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_3_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf28, primals_33, primals_34, primals_35, primals_36, buf29, 49152, grid=grid(49152), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [features_4_conv_0], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 192, 16, 16), (49152, 1, 3072, 192))
        buf31 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [features_4_conv_1, sigmoid_5, mul_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf32, buf30, primals_38, primals_39, primals_40, primals_41, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [features_4_conv_3], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf34 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_4_conv_4, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf29, buf33, primals_43, primals_44, primals_45, primals_46, buf34, 49152, grid=grid(49152), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [features_5_conv_0], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 192, 16, 16), (49152, 1, 3072, 192))
        buf36 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [features_5_conv_1, sigmoid_6, mul_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf37, buf35, primals_48, primals_49, primals_50, primals_51, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [features_5_conv_3], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf39 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_5_conv_4, add_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf34, buf38, primals_53, primals_54, primals_55, primals_56, buf39, 49152, grid=grid(49152), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [features_6_conv_0], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 192, 16, 16), (49152, 1, 3072, 192))
        buf41 = empty_strided_cuda((4, 192, 16, 16), (49152, 1, 3072, 192), torch.float32)
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [features_6_conv_1, sigmoid_7, mul_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_10.run(buf42, buf40, primals_58, primals_59, primals_60, primals_61, 196608, grid=grid(196608), stream=stream0)
        # Topologically Sorted Source Nodes: [features_6_conv_3], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 48, 16, 16), (12288, 1, 768, 48))
        buf44 = empty_strided_cuda((4, 48, 16, 16), (12288, 1, 768, 48), torch.float32)
        # Topologically Sorted Source Nodes: [features_6_conv_4, add_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf39, buf43, primals_63, primals_64, primals_65, primals_66, buf44, 49152, grid=grid(49152), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [features_7_conv_0], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, buf8, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 192, 8, 8), (12288, 1, 1536, 192))
        buf46 = empty_strided_cuda((4, 192, 8, 8), (12288, 1, 1536, 192), torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [features_7_conv_1, sigmoid_8, mul_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_12.run(buf47, buf45, primals_68, primals_69, primals_70, primals_71, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_7_conv_3], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf49 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_7_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf48, primals_73, primals_74, primals_75, primals_76, buf49, 16384, grid=grid(16384), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [features_8_conv_0], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf51 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [features_8_conv_1, sigmoid_9, mul_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf52, buf50, primals_78, primals_79, primals_80, primals_81, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [features_8_conv_3], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf54 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_8_conv_4, add_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf49, buf53, primals_83, primals_84, primals_85, primals_86, buf54, 16384, grid=grid(16384), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [features_9_conv_0], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf56 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [features_9_conv_1, sigmoid_10, mul_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf57, buf55, primals_88, primals_89, primals_90, primals_91, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [features_9_conv_3], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf59 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_9_conv_4, add_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf54, buf58, primals_93, primals_94, primals_95, primals_96, buf59, 16384, grid=grid(16384), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [features_10_conv_0], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf61 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [features_10_conv_1, sigmoid_11, mul_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf62, buf60, primals_98, primals_99, primals_100, primals_101, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [features_10_conv_3], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 64, 8, 8), (4096, 1, 512, 64))
        buf64 = empty_strided_cuda((4, 64, 8, 8), (4096, 1, 512, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_10_conv_4, add_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_15.run(buf59, buf63, primals_103, primals_104, primals_105, primals_106, buf64, 16384, grid=grid(16384), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [features_11_conv_0], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf66 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [features_11_conv_1, sigmoid_12, mul_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_14.run(buf67, buf65, primals_108, primals_109, primals_110, primals_111, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [features_11_conv_3], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_112, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf68, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf69 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_11_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf68, primals_113, primals_114, primals_115, primals_116, buf69, 16384, grid=grid(16384), stream=stream0)
        buf70 = empty_strided_cuda((4, 256, 1, 1), (256, 1, 1024, 1024), torch.float32)
        buf71 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_13, mul_13, features_11_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_17.run(buf71, buf69, 1024, 16, grid=grid(1024), stream=stream0)
        buf72 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_11_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_118, reinterpret_tensor(buf71, (4, 256), (256, 1), 0), reinterpret_tensor(primals_117, (256, 16), (1, 256), 0), alpha=1, beta=1, out=buf72)
        del primals_118
        buf73 = empty_strided_cuda((4, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_14, mul_14], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_18.run(buf72, buf73, 64, grid=grid(64), stream=stream0)
        buf74 = empty_strided_cuda((4, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_11_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_120, buf73, reinterpret_tensor(primals_119, (16, 256), (1, 16), 0), alpha=1, beta=1, out=buf74)
        del primals_120
        buf75 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_13, mul_13, mul_15], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_19.run(buf75, buf74, 16384, grid=grid(16384), stream=stream0)
        # Topologically Sorted Source Nodes: [features_11_conv_7], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf77 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [features_11_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_20.run(buf76, primals_122, primals_123, primals_124, primals_125, buf77, 8192, grid=grid(8192), stream=stream0)
        del primals_125
        # Topologically Sorted Source Nodes: [features_12_conv_0], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf79 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf80 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [features_12_conv_1, sigmoid_17, mul_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf80, buf78, primals_127, primals_128, primals_129, primals_130, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_12_conv_3], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_131, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf81, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf82 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_12_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf81, primals_132, primals_133, primals_134, primals_135, buf82, 32768, grid=grid(32768), stream=stream0)
        buf83 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_19, mul_17, features_12_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf84, buf82, 2048, 16, grid=grid(2048), stream=stream0)
        buf85 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_12_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_137, reinterpret_tensor(buf84, (4, 512), (512, 1), 0), reinterpret_tensor(primals_136, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf85)
        del primals_137
        buf86 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_21, mul_18], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf85, buf86, 128, grid=grid(128), stream=stream0)
        buf87 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_12_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_139, buf86, reinterpret_tensor(primals_138, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf87)
        del primals_139
        buf88 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_19, mul_17, mul_19], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf88, buf87, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_12_conv_7], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf90 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [features_12_conv_8, add_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf77, buf89, primals_141, primals_142, primals_143, primals_144, buf90, 8192, grid=grid(8192), stream=stream0)
        del primals_144
        # Topologically Sorted Source Nodes: [features_13_conv_0], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf92 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf93 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [features_13_conv_1, sigmoid_24, mul_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf93, buf91, primals_146, primals_147, primals_148, primals_149, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_13_conv_3], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_150, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf94, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf95 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_13_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf94, primals_151, primals_152, primals_153, primals_154, buf95, 32768, grid=grid(32768), stream=stream0)
        buf96 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_26, mul_21, features_13_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf97, buf95, 2048, 16, grid=grid(2048), stream=stream0)
        buf98 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_13_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_156, reinterpret_tensor(buf97, (4, 512), (512, 1), 0), reinterpret_tensor(primals_155, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf98)
        del primals_156
        buf99 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_28, mul_22], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf98, buf99, 128, grid=grid(128), stream=stream0)
        buf100 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_13_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_158, buf99, reinterpret_tensor(primals_157, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf100)
        del primals_158
        buf101 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_26, mul_21, mul_23], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf101, buf100, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_13_conv_7], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_159, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf103 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [features_13_conv_8, add_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf90, buf102, primals_160, primals_161, primals_162, primals_163, buf103, 8192, grid=grid(8192), stream=stream0)
        del primals_163
        # Topologically Sorted Source Nodes: [features_14_conv_0], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf105 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf106 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [features_14_conv_1, sigmoid_31, mul_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf106, buf104, primals_165, primals_166, primals_167, primals_168, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_14_conv_3], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf107, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf108 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf107, primals_170, primals_171, primals_172, primals_173, buf108, 32768, grid=grid(32768), stream=stream0)
        buf109 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_33, mul_25, features_14_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf110, buf108, 2048, 16, grid=grid(2048), stream=stream0)
        buf111 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_175, reinterpret_tensor(buf110, (4, 512), (512, 1), 0), reinterpret_tensor(primals_174, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf111)
        del primals_175
        buf112 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_35, mul_26], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf111, buf112, 128, grid=grid(128), stream=stream0)
        buf113 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_177, buf112, reinterpret_tensor(primals_176, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf113)
        del primals_177
        buf114 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_33, mul_25, mul_27], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf114, buf113, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_14_conv_7], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf116 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_8, add_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf103, buf115, primals_179, primals_180, primals_181, primals_182, buf116, 8192, grid=grid(8192), stream=stream0)
        del primals_182
        # Topologically Sorted Source Nodes: [features_15_conv_0], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf118 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [features_15_conv_1, sigmoid_38, mul_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf119, buf117, primals_184, primals_185, primals_186, primals_187, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_15_conv_3], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_188, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf120, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf121 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf120, primals_189, primals_190, primals_191, primals_192, buf121, 32768, grid=grid(32768), stream=stream0)
        buf122 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf123 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_40, mul_29, features_15_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf123, buf121, 2048, 16, grid=grid(2048), stream=stream0)
        buf124 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_194, reinterpret_tensor(buf123, (4, 512), (512, 1), 0), reinterpret_tensor(primals_193, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf124)
        del primals_194
        buf125 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_42, mul_30], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf124, buf125, 128, grid=grid(128), stream=stream0)
        buf126 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_196, buf125, reinterpret_tensor(primals_195, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf126)
        del primals_196
        buf127 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_40, mul_29, mul_31], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf127, buf126, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_15_conv_7], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf129 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_8, add_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf116, buf128, primals_198, primals_199, primals_200, primals_201, buf129, 8192, grid=grid(8192), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [features_16_conv_0], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf131 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [features_16_conv_1, sigmoid_45, mul_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_21.run(buf132, buf130, primals_203, primals_204, primals_205, primals_206, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_16_conv_3], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf133, (4, 512, 4, 4), (8192, 1, 2048, 512))
        buf134 = empty_strided_cuda((4, 512, 4, 4), (8192, 1, 2048, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf133, primals_208, primals_209, primals_210, primals_211, buf134, 32768, grid=grid(32768), stream=stream0)
        buf135 = empty_strided_cuda((4, 512, 1, 1), (512, 1, 2048, 2048), torch.float32)
        buf136 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_47, mul_33, features_16_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_23.run(buf136, buf134, 2048, 16, grid=grid(2048), stream=stream0)
        buf137 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_213, reinterpret_tensor(buf136, (4, 512), (512, 1), 0), reinterpret_tensor(primals_212, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf137)
        del primals_213
        buf138 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_49, mul_34], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf137, buf138, 128, grid=grid(128), stream=stream0)
        buf139 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_215, buf138, reinterpret_tensor(primals_214, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf139)
        del primals_215
        buf140 = buf134; del buf134  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_47, mul_33, mul_35], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_25.run(buf140, buf139, 32768, grid=grid(32768), stream=stream0)
        # Topologically Sorted Source Nodes: [features_16_conv_7], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 128, 4, 4), (2048, 1, 512, 128))
        buf142 = empty_strided_cuda((4, 128, 4, 4), (2048, 1, 512, 128), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_8, add_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_26.run(buf129, buf141, primals_217, primals_218, primals_219, primals_220, buf142, 8192, grid=grid(8192), stream=stream0)
        del primals_220
        # Topologically Sorted Source Nodes: [features_17_conv_0], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf144 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [features_17_conv_1, sigmoid_52, mul_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_27.run(buf145, buf143, primals_222, primals_223, primals_224, primals_225, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_17_conv_3], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf146, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf147 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_28.run(buf146, primals_227, primals_228, primals_229, primals_230, buf147, 49152, grid=grid(49152), stream=stream0)
        buf148 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf149 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_54, mul_37, features_17_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_29.run(buf149, buf147, 3072, 16, grid=grid(3072), stream=stream0)
        buf150 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_232, reinterpret_tensor(buf149, (4, 768), (768, 1), 0), reinterpret_tensor(primals_231, (768, 32), (1, 768), 0), alpha=1, beta=1, out=buf150)
        del primals_232
        buf151 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_56, mul_38], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_24.run(buf150, buf151, 128, grid=grid(128), stream=stream0)
        buf152 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_234, buf151, reinterpret_tensor(primals_233, (32, 768), (1, 32), 0), alpha=1, beta=1, out=buf152)
        del primals_234
        buf153 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_54, mul_37, mul_39], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_30.run(buf153, buf152, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_17_conv_7], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf155 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf154, primals_236, primals_237, primals_238, primals_239, buf155, 10240, grid=grid(10240), stream=stream0)
        del primals_239
        # Topologically Sorted Source Nodes: [features_18_conv_0], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf157 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [features_18_conv_1, sigmoid_59, mul_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf158, buf156, primals_241, primals_242, primals_243, primals_244, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_18_conv_3], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf159, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf160 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf159, primals_246, primals_247, primals_248, primals_249, buf160, 61440, grid=grid(61440), stream=stream0)
        buf161 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf162 = buf161; del buf161  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_61, mul_41, features_18_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf162, buf160, 3840, 16, grid=grid(3840), stream=stream0)
        buf163 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_251, reinterpret_tensor(buf162, (4, 960), (960, 1), 0), reinterpret_tensor(primals_250, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf163)
        del primals_251
        buf164 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_63, mul_42], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf163, buf164, 160, grid=grid(160), stream=stream0)
        buf165 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_253, buf164, reinterpret_tensor(primals_252, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf165)
        del primals_253
        buf166 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_61, mul_41, mul_43], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf166, buf165, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_18_conv_7], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf168 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_8, add_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf155, buf167, primals_255, primals_256, primals_257, primals_258, buf168, 10240, grid=grid(10240), stream=stream0)
        del primals_258
        # Topologically Sorted Source Nodes: [features_19_conv_0], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf170 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [features_19_conv_1, sigmoid_66, mul_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf171, buf169, primals_260, primals_261, primals_262, primals_263, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_19_conv_3], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, primals_264, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf172, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf173 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf172, primals_265, primals_266, primals_267, primals_268, buf173, 61440, grid=grid(61440), stream=stream0)
        buf174 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf175 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_68, mul_45, features_19_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf175, buf173, 3840, 16, grid=grid(3840), stream=stream0)
        buf176 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_270, reinterpret_tensor(buf175, (4, 960), (960, 1), 0), reinterpret_tensor(primals_269, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf176)
        del primals_270
        buf177 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_70, mul_46], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf176, buf177, 160, grid=grid(160), stream=stream0)
        buf178 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_272, buf177, reinterpret_tensor(primals_271, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf178)
        del primals_272
        buf179 = buf173; del buf173  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_68, mul_45, mul_47], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf179, buf178, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_19_conv_7], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf181 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_8, add_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf168, buf180, primals_274, primals_275, primals_276, primals_277, buf181, 10240, grid=grid(10240), stream=stream0)
        del primals_277
        # Topologically Sorted Source Nodes: [features_20_conv_0], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf183 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf184 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [features_20_conv_1, sigmoid_73, mul_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf184, buf182, primals_279, primals_280, primals_281, primals_282, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_20_conv_3], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_283, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf185, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf186 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf185, primals_284, primals_285, primals_286, primals_287, buf186, 61440, grid=grid(61440), stream=stream0)
        buf187 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf188 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_75, mul_49, features_20_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf188, buf186, 3840, 16, grid=grid(3840), stream=stream0)
        buf189 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_289, reinterpret_tensor(buf188, (4, 960), (960, 1), 0), reinterpret_tensor(primals_288, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf189)
        del primals_289
        buf190 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_77, mul_50], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf189, buf190, 160, grid=grid(160), stream=stream0)
        buf191 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_291, buf190, reinterpret_tensor(primals_290, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf191)
        del primals_291
        buf192 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_75, mul_49, mul_51], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf192, buf191, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_20_conv_7], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf194 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_8, add_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf181, buf193, primals_293, primals_294, primals_295, primals_296, buf194, 10240, grid=grid(10240), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [features_21_conv_0], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf196 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [features_21_conv_1, sigmoid_80, mul_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf197, buf195, primals_298, primals_299, primals_300, primals_301, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_3], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf198, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf199 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf198, primals_303, primals_304, primals_305, primals_306, buf199, 61440, grid=grid(61440), stream=stream0)
        buf200 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf201 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_82, mul_53, features_21_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf201, buf199, 3840, 16, grid=grid(3840), stream=stream0)
        buf202 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_308, reinterpret_tensor(buf201, (4, 960), (960, 1), 0), reinterpret_tensor(primals_307, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf202)
        del primals_308
        buf203 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_84, mul_54], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf202, buf203, 160, grid=grid(160), stream=stream0)
        buf204 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_310, buf203, reinterpret_tensor(primals_309, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf204)
        del primals_310
        buf205 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_82, mul_53, mul_55], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf205, buf204, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_7], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf207 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_8, add_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf194, buf206, primals_312, primals_313, primals_314, primals_315, buf207, 10240, grid=grid(10240), stream=stream0)
        del primals_315
        # Topologically Sorted Source Nodes: [features_22_conv_0], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf209 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [features_22_conv_1, sigmoid_87, mul_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf210, buf208, primals_317, primals_318, primals_319, primals_320, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_3], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_321, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf211, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf212 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf211, primals_322, primals_323, primals_324, primals_325, buf212, 61440, grid=grid(61440), stream=stream0)
        buf213 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf214 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_89, mul_57, features_22_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf214, buf212, 3840, 16, grid=grid(3840), stream=stream0)
        buf215 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_327, reinterpret_tensor(buf214, (4, 960), (960, 1), 0), reinterpret_tensor(primals_326, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf215)
        del primals_327
        buf216 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_91, mul_58], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf215, buf216, 160, grid=grid(160), stream=stream0)
        buf217 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_329, buf216, reinterpret_tensor(primals_328, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf217)
        del primals_329
        buf218 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_89, mul_57, mul_59], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf218, buf217, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_7], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_330, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf220 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_8, add_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf207, buf219, primals_331, primals_332, primals_333, primals_334, buf220, 10240, grid=grid(10240), stream=stream0)
        del primals_334
        # Topologically Sorted Source Nodes: [features_23_conv_0], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf222 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [features_23_conv_1, sigmoid_94, mul_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf223, buf221, primals_336, primals_337, primals_338, primals_339, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_3], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, primals_340, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf224, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf225 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf224, primals_341, primals_342, primals_343, primals_344, buf225, 61440, grid=grid(61440), stream=stream0)
        buf226 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf227 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_96, mul_61, features_23_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf227, buf225, 3840, 16, grid=grid(3840), stream=stream0)
        buf228 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_346, reinterpret_tensor(buf227, (4, 960), (960, 1), 0), reinterpret_tensor(primals_345, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf228)
        del primals_346
        buf229 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_98, mul_62], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf228, buf229, 160, grid=grid(160), stream=stream0)
        buf230 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_348, buf229, reinterpret_tensor(primals_347, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf230)
        del primals_348
        buf231 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_96, mul_61, mul_63], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf231, buf230, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_7], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_349, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf233 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_8, add_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf220, buf232, primals_350, primals_351, primals_352, primals_353, buf233, 10240, grid=grid(10240), stream=stream0)
        del primals_353
        # Topologically Sorted Source Nodes: [features_24_conv_0], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_354, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf235 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf236 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [features_24_conv_1, sigmoid_101, mul_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf236, buf234, primals_355, primals_356, primals_357, primals_358, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_3], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_359, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf237, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf238 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf237, primals_360, primals_361, primals_362, primals_363, buf238, 61440, grid=grid(61440), stream=stream0)
        buf239 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf240 = buf239; del buf239  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_103, mul_65, features_24_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf240, buf238, 3840, 16, grid=grid(3840), stream=stream0)
        buf241 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_365, reinterpret_tensor(buf240, (4, 960), (960, 1), 0), reinterpret_tensor(primals_364, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf241)
        del primals_365
        buf242 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_105, mul_66], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf241, buf242, 160, grid=grid(160), stream=stream0)
        buf243 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_367, buf242, reinterpret_tensor(primals_366, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf243)
        del primals_367
        buf244 = buf238; del buf238  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_103, mul_65, mul_67], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf244, buf243, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_7], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_368, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf246 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_8, add_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf233, buf245, primals_369, primals_370, primals_371, primals_372, buf246, 10240, grid=grid(10240), stream=stream0)
        del primals_372
        # Topologically Sorted Source Nodes: [features_25_conv_0], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_373, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf248 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [features_25_conv_1, sigmoid_108, mul_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf249, buf247, primals_374, primals_375, primals_376, primals_377, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_3], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_378, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf250, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf251 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_33.run(buf250, primals_379, primals_380, primals_381, primals_382, buf251, 61440, grid=grid(61440), stream=stream0)
        buf252 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        buf253 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_110, mul_69, features_25_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_34.run(buf253, buf251, 3840, 16, grid=grid(3840), stream=stream0)
        buf254 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_384, reinterpret_tensor(buf253, (4, 960), (960, 1), 0), reinterpret_tensor(primals_383, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf254)
        del primals_384
        buf255 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_112, mul_70], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf254, buf255, 160, grid=grid(160), stream=stream0)
        buf256 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_386, buf255, reinterpret_tensor(primals_385, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf256)
        del primals_386
        buf257 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_110, mul_69, mul_71], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_36.run(buf257, buf256, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_7], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (4, 160, 4, 4), (2560, 1, 640, 160))
        buf259 = empty_strided_cuda((4, 160, 4, 4), (2560, 1, 640, 160), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_8, add_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_37.run(buf246, buf258, primals_388, primals_389, primals_390, primals_391, buf259, 10240, grid=grid(10240), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [features_26_conv_0], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (4, 960, 4, 4), (15360, 1, 3840, 960))
        buf261 = empty_strided_cuda((4, 960, 4, 4), (15360, 1, 3840, 960), torch.float32)
        buf262 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [features_26_conv_1, sigmoid_115, mul_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_32.run(buf262, buf260, primals_393, primals_394, primals_395, primals_396, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_3], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_397, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf263, (4, 960, 2, 2), (3840, 1, 1920, 960))
        buf264 = empty_strided_cuda((4, 960, 2, 2), (3840, 1, 1920, 960), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_38.run(buf263, primals_398, primals_399, primals_400, primals_401, buf264, 15360, grid=grid(15360), stream=stream0)
        buf265 = empty_strided_cuda((4, 960, 1, 1), (960, 1, 3840, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_117, mul_73, features_26_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_39.run(buf264, buf265, 3840, grid=grid(3840), stream=stream0)
        buf266 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_403, reinterpret_tensor(buf265, (4, 960), (960, 1), 0), reinterpret_tensor(primals_402, (960, 40), (1, 960), 0), alpha=1, beta=1, out=buf266)
        del primals_403
        buf267 = empty_strided_cuda((4, 40), (40, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_119, mul_74], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_35.run(buf266, buf267, 160, grid=grid(160), stream=stream0)
        buf268 = empty_strided_cuda((4, 960), (960, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_405, buf267, reinterpret_tensor(primals_404, (40, 960), (1, 40), 0), alpha=1, beta=1, out=buf268)
        del primals_405
        buf269 = buf264; del buf264  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_117, mul_73, mul_75], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_40.run(buf269, buf268, 15360, grid=grid(15360), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_7], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_406, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf271 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf270, primals_407, primals_408, primals_409, primals_410, buf271, 4096, grid=grid(4096), stream=stream0)
        del primals_410
        # Topologically Sorted Source Nodes: [features_27_conv_0], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_411, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf273 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf274 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [features_27_conv_1, sigmoid_122, mul_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf274, buf272, primals_412, primals_413, primals_414, primals_415, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_3], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_416, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf275, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf276 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf275, primals_417, primals_418, primals_419, primals_420, buf276, 24576, grid=grid(24576), stream=stream0)
        buf277 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_124, mul_77, features_27_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf276, buf277, 6144, grid=grid(6144), stream=stream0)
        buf278 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_422, reinterpret_tensor(buf277, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_421, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf278)
        del primals_422
        buf279 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_126, mul_78], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf278, buf279, 256, grid=grid(256), stream=stream0)
        buf280 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_424, buf279, reinterpret_tensor(primals_423, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf280)
        del primals_424
        buf281 = buf276; del buf276  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_124, mul_77, mul_79], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf281, buf280, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_7], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_425, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf283 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_8, add_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf271, buf282, primals_426, primals_427, primals_428, primals_429, buf283, 4096, grid=grid(4096), stream=stream0)
        del primals_429
        # Topologically Sorted Source Nodes: [features_28_conv_0], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_430, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf284, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf285 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf286 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [features_28_conv_1, sigmoid_129, mul_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf286, buf284, primals_431, primals_432, primals_433, primals_434, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_3], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_435, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf287, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf288 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf287, primals_436, primals_437, primals_438, primals_439, buf288, 24576, grid=grid(24576), stream=stream0)
        buf289 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_131, mul_81, features_28_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf288, buf289, 6144, grid=grid(6144), stream=stream0)
        buf290 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_441, reinterpret_tensor(buf289, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_440, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf290)
        del primals_441
        buf291 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_133, mul_82], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf290, buf291, 256, grid=grid(256), stream=stream0)
        buf292 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_443, buf291, reinterpret_tensor(primals_442, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf292)
        del primals_443
        buf293 = buf288; del buf288  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_131, mul_81, mul_83], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf293, buf292, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_7], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_444, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf295 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_8, add_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf283, buf294, primals_445, primals_446, primals_447, primals_448, buf295, 4096, grid=grid(4096), stream=stream0)
        del primals_448
        # Topologically Sorted Source Nodes: [features_29_conv_0], Original ATen: [aten.convolution]
        buf296 = extern_kernels.convolution(buf295, primals_449, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf297 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf298 = buf297; del buf297  # reuse
        # Topologically Sorted Source Nodes: [features_29_conv_1, sigmoid_136, mul_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf298, buf296, primals_450, primals_451, primals_452, primals_453, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_3], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_454, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf299, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf300 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf299, primals_455, primals_456, primals_457, primals_458, buf300, 24576, grid=grid(24576), stream=stream0)
        buf301 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_138, mul_85, features_29_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf300, buf301, 6144, grid=grid(6144), stream=stream0)
        buf302 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_460, reinterpret_tensor(buf301, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_459, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf302)
        del primals_460
        buf303 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_140, mul_86], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf302, buf303, 256, grid=grid(256), stream=stream0)
        buf304 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_462, buf303, reinterpret_tensor(primals_461, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf304)
        del primals_462
        buf305 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_138, mul_85, mul_87], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf305, buf304, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_7], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_463, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf307 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_8, add_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf295, buf306, primals_464, primals_465, primals_466, primals_467, buf307, 4096, grid=grid(4096), stream=stream0)
        del primals_467
        # Topologically Sorted Source Nodes: [features_30_conv_0], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf309 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf310 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [features_30_conv_1, sigmoid_143, mul_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf310, buf308, primals_469, primals_470, primals_471, primals_472, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_3], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_473, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf311, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf312 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf311, primals_474, primals_475, primals_476, primals_477, buf312, 24576, grid=grid(24576), stream=stream0)
        buf313 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_145, mul_89, features_30_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf312, buf313, 6144, grid=grid(6144), stream=stream0)
        buf314 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_479, reinterpret_tensor(buf313, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_478, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf314)
        del primals_479
        buf315 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_147, mul_90], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf314, buf315, 256, grid=grid(256), stream=stream0)
        buf316 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_481, buf315, reinterpret_tensor(primals_480, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf316)
        del primals_481
        buf317 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_145, mul_89, mul_91], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf317, buf316, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_7], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, primals_482, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf319 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_8, add_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf307, buf318, primals_483, primals_484, primals_485, primals_486, buf319, 4096, grid=grid(4096), stream=stream0)
        del primals_486
        # Topologically Sorted Source Nodes: [features_31_conv_0], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_487, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf321 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf322 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [features_31_conv_1, sigmoid_150, mul_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf322, buf320, primals_488, primals_489, primals_490, primals_491, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_3], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_492, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf323, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf324 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf323, primals_493, primals_494, primals_495, primals_496, buf324, 24576, grid=grid(24576), stream=stream0)
        buf325 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_152, mul_93, features_31_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf324, buf325, 6144, grid=grid(6144), stream=stream0)
        buf326 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_498, reinterpret_tensor(buf325, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_497, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf326)
        del primals_498
        buf327 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_154, mul_94], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf326, buf327, 256, grid=grid(256), stream=stream0)
        buf328 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_500, buf327, reinterpret_tensor(primals_499, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf328)
        del primals_500
        buf329 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_152, mul_93, mul_95], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf329, buf328, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_7], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf329, primals_501, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf331 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_8, add_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf319, buf330, primals_502, primals_503, primals_504, primals_505, buf331, 4096, grid=grid(4096), stream=stream0)
        del primals_505
        # Topologically Sorted Source Nodes: [features_32_conv_0], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_506, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf333 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf334 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [features_32_conv_1, sigmoid_157, mul_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf334, buf332, primals_507, primals_508, primals_509, primals_510, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_3], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, primals_511, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf335, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf336 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf335, primals_512, primals_513, primals_514, primals_515, buf336, 24576, grid=grid(24576), stream=stream0)
        buf337 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_159, mul_97, features_32_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf336, buf337, 6144, grid=grid(6144), stream=stream0)
        buf338 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_517, reinterpret_tensor(buf337, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_516, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf338)
        del primals_517
        buf339 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_161, mul_98], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf338, buf339, 256, grid=grid(256), stream=stream0)
        buf340 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_519, buf339, reinterpret_tensor(primals_518, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf340)
        del primals_519
        buf341 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_159, mul_97, mul_99], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf341, buf340, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_7], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_520, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf343 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_8, add_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf331, buf342, primals_521, primals_522, primals_523, primals_524, buf343, 4096, grid=grid(4096), stream=stream0)
        del primals_524
        # Topologically Sorted Source Nodes: [features_33_conv_0], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_525, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf345 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf346 = buf345; del buf345  # reuse
        # Topologically Sorted Source Nodes: [features_33_conv_1, sigmoid_164, mul_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf346, buf344, primals_526, primals_527, primals_528, primals_529, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_3], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_530, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf347, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf348 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf347, primals_531, primals_532, primals_533, primals_534, buf348, 24576, grid=grid(24576), stream=stream0)
        buf349 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_166, mul_101, features_33_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf348, buf349, 6144, grid=grid(6144), stream=stream0)
        buf350 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_536, reinterpret_tensor(buf349, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_535, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf350)
        del primals_536
        buf351 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_168, mul_102], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf350, buf351, 256, grid=grid(256), stream=stream0)
        buf352 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_538, buf351, reinterpret_tensor(primals_537, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf352)
        del primals_538
        buf353 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_166, mul_101, mul_103], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf353, buf352, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_7], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_539, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf355 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_8, add_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf343, buf354, primals_540, primals_541, primals_542, primals_543, buf355, 4096, grid=grid(4096), stream=stream0)
        del primals_543
        # Topologically Sorted Source Nodes: [features_34_conv_0], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_544, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf357 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf358 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [features_34_conv_1, sigmoid_171, mul_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf358, buf356, primals_545, primals_546, primals_547, primals_548, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_3], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_549, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf359, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf360 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf359, primals_550, primals_551, primals_552, primals_553, buf360, 24576, grid=grid(24576), stream=stream0)
        buf361 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_173, mul_105, features_34_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf360, buf361, 6144, grid=grid(6144), stream=stream0)
        buf362 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_555, reinterpret_tensor(buf361, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_554, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf362)
        del primals_555
        buf363 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_175, mul_106], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf362, buf363, 256, grid=grid(256), stream=stream0)
        buf364 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_557, buf363, reinterpret_tensor(primals_556, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf364)
        del primals_557
        buf365 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_173, mul_105, mul_107], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf365, buf364, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_7], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_558, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf367 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_8, add_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf355, buf366, primals_559, primals_560, primals_561, primals_562, buf367, 4096, grid=grid(4096), stream=stream0)
        del primals_562
        # Topologically Sorted Source Nodes: [features_35_conv_0], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, primals_563, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf369 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf370 = buf369; del buf369  # reuse
        # Topologically Sorted Source Nodes: [features_35_conv_1, sigmoid_178, mul_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf370, buf368, primals_564, primals_565, primals_566, primals_567, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_3], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_568, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf371, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf372 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf371, primals_569, primals_570, primals_571, primals_572, buf372, 24576, grid=grid(24576), stream=stream0)
        buf373 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_180, mul_109, features_35_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf372, buf373, 6144, grid=grid(6144), stream=stream0)
        buf374 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_574, reinterpret_tensor(buf373, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_573, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf374)
        del primals_574
        buf375 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_182, mul_110], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf374, buf375, 256, grid=grid(256), stream=stream0)
        buf376 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_576, buf375, reinterpret_tensor(primals_575, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf376)
        del primals_576
        buf377 = buf372; del buf372  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_180, mul_109, mul_111], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf377, buf376, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_7], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_577, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf378, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf379 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_8, add_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf367, buf378, primals_578, primals_579, primals_580, primals_581, buf379, 4096, grid=grid(4096), stream=stream0)
        del primals_581
        # Topologically Sorted Source Nodes: [features_36_conv_0], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_582, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf381 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf382 = buf381; del buf381  # reuse
        # Topologically Sorted Source Nodes: [features_36_conv_1, sigmoid_185, mul_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf382, buf380, primals_583, primals_584, primals_585, primals_586, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_3], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_587, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf383, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf384 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf383, primals_588, primals_589, primals_590, primals_591, buf384, 24576, grid=grid(24576), stream=stream0)
        buf385 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_187, mul_113, features_36_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf384, buf385, 6144, grid=grid(6144), stream=stream0)
        buf386 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_593, reinterpret_tensor(buf385, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_592, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf386)
        del primals_593
        buf387 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_189, mul_114], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf386, buf387, 256, grid=grid(256), stream=stream0)
        buf388 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_595, buf387, reinterpret_tensor(primals_594, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf388)
        del primals_595
        buf389 = buf384; del buf384  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_187, mul_113, mul_115], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf389, buf388, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_7], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, primals_596, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf391 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_8, add_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf379, buf390, primals_597, primals_598, primals_599, primals_600, buf391, 4096, grid=grid(4096), stream=stream0)
        del primals_600
        # Topologically Sorted Source Nodes: [features_37_conv_0], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_601, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf393 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf394 = buf393; del buf393  # reuse
        # Topologically Sorted Source Nodes: [features_37_conv_1, sigmoid_192, mul_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf394, buf392, primals_602, primals_603, primals_604, primals_605, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_3], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_606, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf395, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf396 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf395, primals_607, primals_608, primals_609, primals_610, buf396, 24576, grid=grid(24576), stream=stream0)
        buf397 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_194, mul_117, features_37_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf396, buf397, 6144, grid=grid(6144), stream=stream0)
        buf398 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_612, reinterpret_tensor(buf397, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_611, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf398)
        del primals_612
        buf399 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_196, mul_118], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf398, buf399, 256, grid=grid(256), stream=stream0)
        buf400 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_614, buf399, reinterpret_tensor(primals_613, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf400)
        del primals_614
        buf401 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_194, mul_117, mul_119], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf401, buf400, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_7], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_615, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf403 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_8, add_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf391, buf402, primals_616, primals_617, primals_618, primals_619, buf403, 4096, grid=grid(4096), stream=stream0)
        del primals_619
        # Topologically Sorted Source Nodes: [features_38_conv_0], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_620, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf405 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf406 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [features_38_conv_1, sigmoid_199, mul_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf406, buf404, primals_621, primals_622, primals_623, primals_624, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_3], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, primals_625, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf407, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf408 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf407, primals_626, primals_627, primals_628, primals_629, buf408, 24576, grid=grid(24576), stream=stream0)
        buf409 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_201, mul_121, features_38_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf408, buf409, 6144, grid=grid(6144), stream=stream0)
        buf410 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_631, reinterpret_tensor(buf409, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_630, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf410)
        del primals_631
        buf411 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_203, mul_122], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf410, buf411, 256, grid=grid(256), stream=stream0)
        buf412 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_633, buf411, reinterpret_tensor(primals_632, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf412)
        del primals_633
        buf413 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_201, mul_121, mul_123], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf413, buf412, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_7], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_634, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf414, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf415 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_8, add_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf403, buf414, primals_635, primals_636, primals_637, primals_638, buf415, 4096, grid=grid(4096), stream=stream0)
        del primals_638
        # Topologically Sorted Source Nodes: [features_39_conv_0], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_639, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf417 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf418 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [features_39_conv_1, sigmoid_206, mul_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf418, buf416, primals_640, primals_641, primals_642, primals_643, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_3], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_644, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf419, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf420 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf419, primals_645, primals_646, primals_647, primals_648, buf420, 24576, grid=grid(24576), stream=stream0)
        buf421 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_208, mul_125, features_39_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf420, buf421, 6144, grid=grid(6144), stream=stream0)
        buf422 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_650, reinterpret_tensor(buf421, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_649, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf422)
        del primals_650
        buf423 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_210, mul_126], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf422, buf423, 256, grid=grid(256), stream=stream0)
        buf424 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_652, buf423, reinterpret_tensor(primals_651, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf424)
        del primals_652
        buf425 = buf420; del buf420  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_208, mul_125, mul_127], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf425, buf424, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_7], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_653, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf427 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_8, add_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf415, buf426, primals_654, primals_655, primals_656, primals_657, buf427, 4096, grid=grid(4096), stream=stream0)
        del primals_657
        # Topologically Sorted Source Nodes: [features_40_conv_0], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_658, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf428, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf429 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        buf430 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [features_40_conv_1, sigmoid_213, mul_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_42.run(buf430, buf428, primals_659, primals_660, primals_661, primals_662, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_3], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_663, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf431, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf432 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_43.run(buf431, primals_664, primals_665, primals_666, primals_667, buf432, 24576, grid=grid(24576), stream=stream0)
        buf433 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_215, mul_129, features_40_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_44.run(buf432, buf433, 6144, grid=grid(6144), stream=stream0)
        buf434 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_669, reinterpret_tensor(buf433, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_668, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf434)
        del primals_669
        buf435 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_217, mul_130], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_45.run(buf434, buf435, 256, grid=grid(256), stream=stream0)
        buf436 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_671, buf435, reinterpret_tensor(primals_670, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf436)
        del primals_671
        buf437 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_215, mul_129, mul_131], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_46.run(buf437, buf436, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_7], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_672, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (4, 256, 2, 2), (1024, 1, 512, 256))
        buf439 = empty_strided_cuda((4, 256, 2, 2), (1024, 1, 512, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_8, add_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_47.run(buf427, buf438, primals_673, primals_674, primals_675, primals_676, buf439, 4096, grid=grid(4096), stream=stream0)
        del primals_676
        # Topologically Sorted Source Nodes: [conv_0], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, primals_677, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (4, 1792, 2, 2), (7168, 1, 3584, 1792))
        buf441 = empty_strided_cuda((4, 1792, 2, 2), (7168, 1, 3584, 1792), torch.float32)
        # Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_48.run(buf440, primals_678, primals_679, primals_680, primals_681, buf441, 28672, grid=grid(28672), stream=stream0)
        buf442 = empty_strided_cuda((4, 1792, 1, 1), (1792, 1, 7168, 7168), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_220, mul_132, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_49.run(buf441, buf442, 7168, grid=grid(7168), stream=stream0)
        del buf441
        buf443 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [classifier], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_683, reinterpret_tensor(buf442, (4, 1792), (1792, 1), 0), reinterpret_tensor(primals_682, (1792, 1000), (1, 1792), 0), alpha=1, beta=1, out=buf443)
        del primals_683
    return (buf443, buf0, buf1, primals_3, primals_4, primals_5, primals_6, buf2, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, buf3, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, buf4, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, buf5, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, buf6, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, buf7, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, buf8, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, buf9, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, buf10, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, buf11, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_121, primals_122, primals_123, primals_124, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_140, primals_141, primals_142, primals_143, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_159, primals_160, primals_161, primals_162, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_216, primals_217, primals_218, primals_219, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_235, primals_236, primals_237, primals_238, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_254, primals_255, primals_256, primals_257, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_273, primals_274, primals_275, primals_276, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_311, primals_312, primals_313, primals_314, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_330, primals_331, primals_332, primals_333, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_349, primals_350, primals_351, primals_352, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_368, primals_369, primals_370, primals_371, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_406, primals_407, primals_408, primals_409, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_425, primals_426, primals_427, primals_428, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_444, primals_445, primals_446, primals_447, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_463, primals_464, primals_465, primals_466, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_482, primals_483, primals_484, primals_485, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_501, primals_502, primals_503, primals_504, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_520, primals_521, primals_522, primals_523, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_539, primals_540, primals_541, primals_542, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_558, primals_559, primals_560, primals_561, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_577, primals_578, primals_579, primals_580, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_596, primals_597, primals_598, primals_599, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_615, primals_616, primals_617, primals_618, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_634, primals_635, primals_636, primals_637, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_653, primals_654, primals_655, primals_656, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_672, primals_673, primals_674, primals_675, primals_677, primals_678, primals_679, primals_680, primals_681, buf12, buf14, buf15, buf17, buf18, buf19, buf20, buf22, buf23, buf24, buf25, buf27, buf28, buf29, buf30, buf32, buf33, buf34, buf35, buf37, buf38, buf39, buf40, buf42, buf43, buf44, buf45, buf47, buf48, buf49, buf50, buf52, buf53, buf54, buf55, buf57, buf58, buf59, buf60, buf62, buf63, buf64, buf65, buf67, buf68, reinterpret_tensor(buf71, (4, 256), (256, 1), 0), buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf80, buf81, reinterpret_tensor(buf84, (4, 512), (512, 1), 0), buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf93, buf94, reinterpret_tensor(buf97, (4, 512), (512, 1), 0), buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf106, buf107, reinterpret_tensor(buf110, (4, 512), (512, 1), 0), buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf119, buf120, reinterpret_tensor(buf123, (4, 512), (512, 1), 0), buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf132, buf133, reinterpret_tensor(buf136, (4, 512), (512, 1), 0), buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf145, buf146, reinterpret_tensor(buf149, (4, 768), (768, 1), 0), buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf158, buf159, reinterpret_tensor(buf162, (4, 960), (960, 1), 0), buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf171, buf172, reinterpret_tensor(buf175, (4, 960), (960, 1), 0), buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf184, buf185, reinterpret_tensor(buf188, (4, 960), (960, 1), 0), buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf197, buf198, reinterpret_tensor(buf201, (4, 960), (960, 1), 0), buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf210, buf211, reinterpret_tensor(buf214, (4, 960), (960, 1), 0), buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf223, buf224, reinterpret_tensor(buf227, (4, 960), (960, 1), 0), buf228, buf229, buf230, buf231, buf232, buf233, buf234, buf236, buf237, reinterpret_tensor(buf240, (4, 960), (960, 1), 0), buf241, buf242, buf243, buf244, buf245, buf246, buf247, buf249, buf250, reinterpret_tensor(buf253, (4, 960), (960, 1), 0), buf254, buf255, buf256, buf257, buf258, buf259, buf260, buf262, buf263, reinterpret_tensor(buf265, (4, 960), (960, 1), 0), buf266, buf267, buf268, buf269, buf270, buf271, buf272, buf274, buf275, reinterpret_tensor(buf277, (4, 1536), (1536, 1), 0), buf278, buf279, buf280, buf281, buf282, buf283, buf284, buf286, buf287, reinterpret_tensor(buf289, (4, 1536), (1536, 1), 0), buf290, buf291, buf292, buf293, buf294, buf295, buf296, buf298, buf299, reinterpret_tensor(buf301, (4, 1536), (1536, 1), 0), buf302, buf303, buf304, buf305, buf306, buf307, buf308, buf310, buf311, reinterpret_tensor(buf313, (4, 1536), (1536, 1), 0), buf314, buf315, buf316, buf317, buf318, buf319, buf320, buf322, buf323, reinterpret_tensor(buf325, (4, 1536), (1536, 1), 0), buf326, buf327, buf328, buf329, buf330, buf331, buf332, buf334, buf335, reinterpret_tensor(buf337, (4, 1536), (1536, 1), 0), buf338, buf339, buf340, buf341, buf342, buf343, buf344, buf346, buf347, reinterpret_tensor(buf349, (4, 1536), (1536, 1), 0), buf350, buf351, buf352, buf353, buf354, buf355, buf356, buf358, buf359, reinterpret_tensor(buf361, (4, 1536), (1536, 1), 0), buf362, buf363, buf364, buf365, buf366, buf367, buf368, buf370, buf371, reinterpret_tensor(buf373, (4, 1536), (1536, 1), 0), buf374, buf375, buf376, buf377, buf378, buf379, buf380, buf382, buf383, reinterpret_tensor(buf385, (4, 1536), (1536, 1), 0), buf386, buf387, buf388, buf389, buf390, buf391, buf392, buf394, buf395, reinterpret_tensor(buf397, (4, 1536), (1536, 1), 0), buf398, buf399, buf400, buf401, buf402, buf403, buf404, buf406, buf407, reinterpret_tensor(buf409, (4, 1536), (1536, 1), 0), buf410, buf411, buf412, buf413, buf414, buf415, buf416, buf418, buf419, reinterpret_tensor(buf421, (4, 1536), (1536, 1), 0), buf422, buf423, buf424, buf425, buf426, buf427, buf428, buf430, buf431, reinterpret_tensor(buf433, (4, 1536), (1536, 1), 0), buf434, buf435, buf436, buf437, buf438, buf439, buf440, reinterpret_tensor(buf442, (4, 1792), (1792, 1), 0), primals_682, primals_670, primals_668, primals_651, primals_649, primals_632, primals_630, primals_613, primals_611, primals_594, primals_592, primals_575, primals_573, primals_556, primals_554, primals_537, primals_535, primals_518, primals_516, primals_499, primals_497, primals_480, primals_478, primals_461, primals_459, primals_442, primals_440, primals_423, primals_421, primals_404, primals_402, primals_385, primals_383, primals_366, primals_364, primals_347, primals_345, primals_328, primals_326, primals_309, primals_307, primals_290, primals_288, primals_271, primals_269, primals_252, primals_250, primals_233, primals_231, primals_214, primals_212, primals_195, primals_193, primals_176, primals_174, primals_157, primals_155, primals_138, primals_136, primals_119, primals_117, )


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
    primals_27 = rand_strided((96, 24, 3, 3), (216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((48, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_72 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((16, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((960, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((40, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((960, 40), (40, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((256, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((1792, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((1000, 1792), (1792, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
