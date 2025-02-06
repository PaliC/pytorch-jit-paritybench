# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/aa/caaphy6jmftxbj3ckdql3butbwcqp5ayaf2ivjs3siifg2jq4ar4.py
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
    size_hints={'y': 1024, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7p/c7phjsrzmxsfsizlvfrkfu6e4ory65fbjpseypbewsopbvocbovu.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6t/c6t7iiwsuwvt4yxwjefywtd7etgaagwe7oug6bsv2cxgk4bul4eg.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/md/cmdw6v7mwiokxcs5xnf5qakjfv2w6g3l37lxi462mvpzey72xtsa.py
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
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, %sigmoid), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wx/cwxyfqlutm5mb3nuh25ydkrgahcqxhurzdxbqelr27xs4ndt7xfq.py
# Topologically Sorted Source Nodes: [features_1_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_1_conv_4 => add_5, mul_10, mul_9, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %unsqueeze_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
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


# kernel path: inductor_cache/2p/c2p7ohxmhyhsvtpcwiynghqfe5ckq52zmf5rhkihwy4u4txqzjai.py
# Topologically Sorted Source Nodes: [features_2_conv_1, sigmoid_3, mul_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_2_conv_1 => add_7, mul_12, mul_13, sub_3
#   mul_3 => mul_14
#   sigmoid_3 => sigmoid_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_31), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_7,), kwargs = {})
#   %mul_14 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %sigmoid_2), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
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


# kernel path: inductor_cache/up/cupfb2dxmq7g4skgflrhtm4msqvfe42y6sq34rvtu6rytqxtnlxz.py
# Topologically Sorted Source Nodes: [features_2_conv_4, add_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_1 => add_10
#   features_2_conv_4 => add_9, mul_16, mul_17, sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_39), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %add_9), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
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


# kernel path: inductor_cache/7q/c7qsvp2q74vdj5ukaddcgiwqlmbb2vaseznrnycugueooudtmzgh.py
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
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
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
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/jo/cjojf3rfs7sv6iket7unae2sgga3zehx4uopqqs3z54t45ebfjrr.py
# Topologically Sorted Source Nodes: [features_5_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_5_conv_4 => add_24, mul_37, mul_38, sub_10
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_85), kwargs = {})
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_87), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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


# kernel path: inductor_cache/ir/cirz3gtmphw2j2jdafgpohgrtbnwdq7bwrjfscxlz5mkpowu2kft.py
# Topologically Sorted Source Nodes: [features_6_conv_1, sigmoid_7, mul_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_6_conv_1 => add_26, mul_40, mul_41, sub_11
#   mul_7 => mul_42
#   sigmoid_7 => sigmoid_6
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_95), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_26,), kwargs = {})
#   %mul_42 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_26, %sigmoid_6), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/zq/czqlnlaqacyiyvulx3cwrs5ez5dolhby5kmjqmpm3tnc53cxb34i.py
# Topologically Sorted Source Nodes: [features_6_conv_4, add_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_4 => add_29
#   features_6_conv_4 => add_28, mul_44, mul_45, sub_12
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_97), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_99), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_44, %unsqueeze_101), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_45, %unsqueeze_103), kwargs = {})
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_24, %add_28), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
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


# kernel path: inductor_cache/tq/ctqeii2agkpp6gglnovrhnswlyjy6t2vw7neemghp3hngrk32zzl.py
# Topologically Sorted Source Nodes: [features_12_conv_1, sigmoid_13, mul_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_12_conv_1 => add_56, mul_82, mul_83, sub_23
#   mul_13 => mul_84
#   sigmoid_13 => sigmoid_12
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_189), kwargs = {})
#   %add_56 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_191), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_56,), kwargs = {})
#   %mul_84 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_56, %sigmoid_12), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/eq/ceq53byjpx4mwgxhyen7zdgloxc6h2yuvuziyfnazmjg2jyvfkpp.py
# Topologically Sorted Source Nodes: [features_12_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_12_conv_4 => add_58, mul_86, mul_87, sub_24
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_86, %unsqueeze_197), kwargs = {})
#   %add_58 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_87, %unsqueeze_199), kwargs = {})
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
    xnumel = 24576
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/wj/cwjbkzqq2gpntopx5in2eajtuvhuqqywysvzpgs44qjn2rgrozf2.py
# Topologically Sorted Source Nodes: [features_13_conv_1, sigmoid_14, mul_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_13_conv_1 => add_60, mul_89, mul_90, sub_25
#   mul_14 => mul_91
#   sigmoid_14 => sigmoid_13
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %unsqueeze_205), kwargs = {})
#   %add_60 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %unsqueeze_207), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_60,), kwargs = {})
#   %mul_91 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_60, %sigmoid_13), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
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


# kernel path: inductor_cache/cc/cccyxzkhfkiyahurjfywtx3cpdektcaum576lrx7nqe3wwk4lszs.py
# Topologically Sorted Source Nodes: [features_13_conv_4, add_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_10 => add_63
#   features_13_conv_4 => add_62, mul_93, mul_94, sub_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %unsqueeze_213), kwargs = {})
#   %add_62 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %unsqueeze_215), kwargs = {})
#   %add_63 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_58, %add_62), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 96)
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


# kernel path: inductor_cache/3x/c3xomg5tcopwg2tkuwjxqhfmgswyhwn2xj5cigelupzgyoeydhdp.py
# Topologically Sorted Source Nodes: [features_19_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_19_conv_4 => add_92, mul_135, mul_136, sub_38
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_38, %unsqueeze_305), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_136 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_135, %unsqueeze_309), kwargs = {})
#   %add_92 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_136, %unsqueeze_311), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 384)
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


# kernel path: inductor_cache/k2/ck2pdqbdenypdhiwtofxdq2bi32j3l4i7b6g2u323me3yorqpvlp.py
# Topologically Sorted Source Nodes: [sigmoid_21, mul_21, features_19_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_19_conv_6_avg_pool => mean
#   mul_21 => mul_137
#   sigmoid_21 => sigmoid_20
# Graph fragment:
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_92,), kwargs = {})
#   %mul_137 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %sigmoid_20), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_137, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_20 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_20', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_20(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 384)
    x1 = xindex // 384
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*r2 + 6144*x1), xmask, other=0.0)
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


# kernel path: inductor_cache/5b/c5b3ex5kcq5r3d6ghd27hkrk4nuaoedtpnchwvjs3oljg25b2aio.py
# Topologically Sorted Source Nodes: [sigmoid_22, mul_22], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_22 => mul_138
#   sigmoid_22 => sigmoid_21
# Graph fragment:
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm,), kwargs = {})
#   %mul_138 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, %sigmoid_21), kwargs = {})
triton_poi_fused_mul_sigmoid_21 = async_compile.triton('triton_poi_fused_mul_sigmoid_21', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/q7/cq7ny54u63te6p4nqskl5xbudlfh4tntugmajf6pmzckdp4omkud.py
# Topologically Sorted Source Nodes: [sigmoid_21, mul_21, mul_23], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_21 => mul_137
#   mul_23 => mul_139
#   sigmoid_21 => sigmoid_20
# Graph fragment:
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_92,), kwargs = {})
#   %mul_137 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_92, %sigmoid_20), kwargs = {})
#   %mul_139 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, %view_1), kwargs = {})
triton_poi_fused_mul_sigmoid_22 = async_compile.triton('triton_poi_fused_mul_sigmoid_22', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 384)
    x2 = xindex // 6144
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 384*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/px/cpxdxcmvjn6xdfmc6h6x6mjpyx2q6tm7khgi2zvmvn2pb27oiaq3.py
# Topologically Sorted Source Nodes: [features_19_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_19_conv_8 => add_94, mul_141, mul_142, sub_39
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_142 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_141, %unsqueeze_317), kwargs = {})
#   %add_94 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_142, %unsqueeze_319), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/d3/cd3qdntuu5rxk3u2u65qxuw54wqtt4yxs2gr2ni43npv7eoygjko.py
# Topologically Sorted Source Nodes: [features_20_conv_1, sigmoid_25, mul_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_20_conv_1 => add_96, mul_144, mul_145, sub_40
#   mul_24 => mul_146
#   sigmoid_25 => sigmoid_23
# Graph fragment:
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_321), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_144, %unsqueeze_325), kwargs = {})
#   %add_96 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_145, %unsqueeze_327), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_96,), kwargs = {})
#   %mul_146 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_96, %sigmoid_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xg/cxgzgadhcyp3ubyyi3qm7a6jiflz5kuaq42vd2tewausvhicavpi.py
# Topologically Sorted Source Nodes: [features_20_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_20_conv_4 => add_98, mul_148, mul_149, sub_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_148 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_148, %unsqueeze_333), kwargs = {})
#   %add_98 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_149, %unsqueeze_335), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/2h/c2hcr4zsjccq6wbqxxmxgcmahahclhrnkzaasoppyxngxxs7tqvl.py
# Topologically Sorted Source Nodes: [sigmoid_27, mul_25, features_20_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_20_conv_6_avg_pool => mean_1
#   mul_25 => mul_150
#   sigmoid_27 => sigmoid_24
# Graph fragment:
#   %sigmoid_24 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_98,), kwargs = {})
#   %mul_150 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_98, %sigmoid_24), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_150, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_26 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_26', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_26', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_26(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/uw/cuw4apc4iw7gmbwxaijfqqntceshvfuxoa7d2wfmoaq3mzzkf4ro.py
# Topologically Sorted Source Nodes: [sigmoid_29, mul_26], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_26 => mul_151
#   sigmoid_29 => sigmoid_25
# Graph fragment:
#   %sigmoid_25 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_2,), kwargs = {})
#   %mul_151 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_2, %sigmoid_25), kwargs = {})
triton_poi_fused_mul_sigmoid_27 = async_compile.triton('triton_poi_fused_mul_sigmoid_27', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wt/cwtbdpgahvsshfn3c2j6bblk3f2ixuso7vvtx4wlqighav7nijr6.py
# Topologically Sorted Source Nodes: [sigmoid_27, mul_25, mul_27], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_25 => mul_150
#   mul_27 => mul_152
#   sigmoid_27 => sigmoid_24
# Graph fragment:
#   %sigmoid_24 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_98,), kwargs = {})
#   %mul_150 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_98, %sigmoid_24), kwargs = {})
#   %mul_152 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_150, %view_3), kwargs = {})
triton_poi_fused_mul_sigmoid_28 = async_compile.triton('triton_poi_fused_mul_sigmoid_28', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_28(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5k/c5kvxelp4azlv2h4fawam3dqo2h6tojksy3hxmsviwwylpadjtzi.py
# Topologically Sorted Source Nodes: [features_20_conv_8, add_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_16 => add_101
#   features_20_conv_8 => add_100, mul_154, mul_155, sub_42
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_154 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_154, %unsqueeze_341), kwargs = {})
#   %add_100 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_155, %unsqueeze_343), kwargs = {})
#   %add_101 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_94, %add_100), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_29 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_29', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
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


# kernel path: inductor_cache/im/cimxa3tpdq7mmeguwg27t22zqqcw7v23zflt6idopaehc7rju7kr.py
# Topologically Sorted Source Nodes: [features_29_conv_1, sigmoid_88, mul_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_29_conv_1 => add_159, mul_261, mul_262, sub_67
#   mul_60 => mul_263
#   sigmoid_88 => sigmoid_59
# Graph fragment:
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_261 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_262 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_261, %unsqueeze_541), kwargs = {})
#   %add_159 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_262, %unsqueeze_543), kwargs = {})
#   %sigmoid_59 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_159,), kwargs = {})
#   %mul_263 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_159, %sigmoid_59), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_30 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_30', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1152)
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


# kernel path: inductor_cache/sy/csyi4dwrx64vfgnuswbywo2uxcrwtsuqwuhtjgw2upd7sltyot2a.py
# Topologically Sorted Source Nodes: [features_29_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_29_conv_4 => add_161, mul_265, mul_266, sub_68
# Graph fragment:
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_265 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_265, %unsqueeze_549), kwargs = {})
#   %add_161 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_266, %unsqueeze_551), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_31', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1152)
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


# kernel path: inductor_cache/2b/c2bpgsjsaevxo5rzbx2hrnkudz6fopp5geglejmlgnmzhy4snxej.py
# Topologically Sorted Source Nodes: [sigmoid_90, mul_61, features_29_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_29_conv_6_avg_pool => mean_10
#   mul_61 => mul_267
#   sigmoid_90 => sigmoid_60
# Graph fragment:
#   %sigmoid_60 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_161,), kwargs = {})
#   %mul_267 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %sigmoid_60), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_267, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_32 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_32', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_32(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4608
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 1152)
    x1 = xindex // 1152
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1152*r2 + 18432*x1), xmask, other=0.0)
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


# kernel path: inductor_cache/bl/cblpgwusxotrtxp3qufrn4chvyv35b22q4ikthheidjvcmidcats.py
# Topologically Sorted Source Nodes: [sigmoid_90, mul_61, mul_63], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_61 => mul_267
#   mul_63 => mul_269
#   sigmoid_90 => sigmoid_60
# Graph fragment:
#   %sigmoid_60 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_161,), kwargs = {})
#   %mul_267 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_161, %sigmoid_60), kwargs = {})
#   %mul_269 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_267, %view_21), kwargs = {})
triton_poi_fused_mul_sigmoid_33 = async_compile.triton('triton_poi_fused_mul_sigmoid_33', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_33(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1152)
    x2 = xindex // 18432
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 1152*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/5d/c5d552j63oxoxuvzgdglrmqdwok4ffhaiv3zc5rmssdz4ghcs2h5.py
# Topologically Sorted Source Nodes: [features_29_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_29_conv_8 => add_163, mul_271, mul_272, sub_69
# Graph fragment:
#   %sub_69 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_553), kwargs = {})
#   %mul_271 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_69, %unsqueeze_555), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_271, %unsqueeze_557), kwargs = {})
#   %add_163 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_272, %unsqueeze_559), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_34', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 224)
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


# kernel path: inductor_cache/h4/ch4c3zz3qmckug6ncbessmfn5dxqkdmlqx5jg6yyib3g7uxl4esi.py
# Topologically Sorted Source Nodes: [features_30_conv_1, sigmoid_95, mul_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_30_conv_1 => add_165, mul_274, mul_275, sub_70
#   mul_64 => mul_276
#   sigmoid_95 => sigmoid_63
# Graph fragment:
#   %sub_70 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_70, %unsqueeze_561), kwargs = {})
#   %mul_274 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_70, %unsqueeze_563), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_274, %unsqueeze_565), kwargs = {})
#   %add_165 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_275, %unsqueeze_567), kwargs = {})
#   %sigmoid_63 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_165,), kwargs = {})
#   %mul_276 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_165, %sigmoid_63), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 86016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1344)
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


# kernel path: inductor_cache/7m/c7mwjjpvijwp3zi6sdfco6tm76ku2qc5mitax45u3qm5jjhdf5co.py
# Topologically Sorted Source Nodes: [features_30_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_30_conv_4 => add_167, mul_278, mul_279, sub_71
# Graph fragment:
#   %sub_71 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_569), kwargs = {})
#   %mul_278 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_71, %unsqueeze_571), kwargs = {})
#   %mul_279 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_278, %unsqueeze_573), kwargs = {})
#   %add_167 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_279, %unsqueeze_575), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 86016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1344)
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


# kernel path: inductor_cache/h6/ch62vyzdyyfopwq4w4rag4sxotvpxnq2a2u6roclrgatl2eygoyw.py
# Topologically Sorted Source Nodes: [sigmoid_97, mul_65, features_30_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_30_conv_6_avg_pool => mean_11
#   mul_65 => mul_280
#   sigmoid_97 => sigmoid_64
# Graph fragment:
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_167,), kwargs = {})
#   %mul_280 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_167, %sigmoid_64), kwargs = {})
#   %mean_11 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_280, [-1, -2], True), kwargs = {})
triton_per_fused_mean_mul_sigmoid_37 = async_compile.triton('triton_per_fused_mean_mul_sigmoid_37', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_mul_sigmoid_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_mul_sigmoid_37(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 1344)
    x1 = xindex // 1344
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1344*r2 + 21504*x1), xmask, other=0.0)
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


# kernel path: inductor_cache/ne/cne2hpcu74bhe7bxe65w742qwoo5agjehfuescjbejnq4c6d3hu6.py
# Topologically Sorted Source Nodes: [sigmoid_99, mul_66], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_66 => mul_281
#   sigmoid_99 => sigmoid_65
# Graph fragment:
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_22,), kwargs = {})
#   %mul_281 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_22, %sigmoid_65), kwargs = {})
triton_poi_fused_mul_sigmoid_38 = async_compile.triton('triton_poi_fused_mul_sigmoid_38', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_38(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/id/cidulxlvkcntturzewwt4zwqxzep5hpfxvzxwm4rj5axca3n7jah.py
# Topologically Sorted Source Nodes: [sigmoid_97, mul_65, mul_67], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_65 => mul_280
#   mul_67 => mul_282
#   sigmoid_97 => sigmoid_64
# Graph fragment:
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_167,), kwargs = {})
#   %mul_280 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_167, %sigmoid_64), kwargs = {})
#   %mul_282 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_280, %view_23), kwargs = {})
triton_poi_fused_mul_sigmoid_39 = async_compile.triton('triton_poi_fused_mul_sigmoid_39', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_39(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 86016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1344)
    x2 = xindex // 21504
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 1344*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/wp/cwp3h6btwbmwxru2qzxjvh4rya77fodhsh7ropjbbgeu35ibr24d.py
# Topologically Sorted Source Nodes: [features_30_conv_8, add_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_25 => add_170
#   features_30_conv_8 => add_169, mul_284, mul_285, sub_72
# Graph fragment:
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %unsqueeze_581), kwargs = {})
#   %add_169 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_285, %unsqueeze_583), kwargs = {})
#   %add_170 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_163, %add_169), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_40', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_40', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 224)
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


# kernel path: inductor_cache/yq/cyqy3jrcfugjstior43cvyg5heoubxeblgdk2rwhwdlde3d4ch6f.py
# Topologically Sorted Source Nodes: [features_48_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_48_conv_4 => add_293, mul_512, mul_513, sub_125
# Graph fragment:
#   %sub_125 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_1001), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_125, %unsqueeze_1003), kwargs = {})
#   %mul_513 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_512, %unsqueeze_1005), kwargs = {})
#   %add_293 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_513, %unsqueeze_1007), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_41', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_41', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1344)
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


# kernel path: inductor_cache/wy/cwysilydb66jusofgp6fhxzdq7lwr63woehwpexuqb4aqdoeoua7.py
# Topologically Sorted Source Nodes: [sigmoid_223, mul_137, features_48_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_48_conv_6_avg_pool => mean_29
#   mul_137 => mul_514
#   sigmoid_223 => sigmoid_136
# Graph fragment:
#   %sigmoid_136 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_293,), kwargs = {})
#   %mul_514 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_293, %sigmoid_136), kwargs = {})
#   %mean_29 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_514, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_42 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_42', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_42', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_42(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1344)
    x1 = xindex // 1344
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 5376*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (1344 + x0 + 5376*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (2688 + x0 + 5376*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (4032 + x0 + 5376*x1), xmask)
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


# kernel path: inductor_cache/5g/c5gk4e5u4fhyqpte3uirduisuvzigbcc2dcevimus4km7lt32ope.py
# Topologically Sorted Source Nodes: [sigmoid_223, mul_137, mul_139], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_137 => mul_514
#   mul_139 => mul_516
#   sigmoid_223 => sigmoid_136
# Graph fragment:
#   %sigmoid_136 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_293,), kwargs = {})
#   %mul_514 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_293, %sigmoid_136), kwargs = {})
#   %mul_516 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_514, %view_59), kwargs = {})
triton_poi_fused_mul_sigmoid_43 = async_compile.triton('triton_poi_fused_mul_sigmoid_43', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_43(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 1344)
    x2 = xindex // 5376
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp3 = tl.load(in_ptr0 + (x0 + 1344*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hs/chsl5e4b3sq76zl2dd2nmijeen465ill76by2gnmvuxsvl7tpvwl.py
# Topologically Sorted Source Nodes: [features_48_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_48_conv_8 => add_295, mul_518, mul_519, sub_126
# Graph fragment:
#   %sub_126 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_126, %unsqueeze_1009), kwargs = {})
#   %mul_518 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_126, %unsqueeze_1011), kwargs = {})
#   %mul_519 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_518, %unsqueeze_1013), kwargs = {})
#   %add_295 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_519, %unsqueeze_1015), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_44', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_44', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
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


# kernel path: inductor_cache/zv/czv7dwvkfa3gey2skl7yuqvqnwinjrz44iucemik72qinzx74gwj.py
# Topologically Sorted Source Nodes: [features_49_conv_1, sigmoid_228, mul_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_49_conv_1 => add_297, mul_521, mul_522, sub_127
#   mul_140 => mul_523
#   sigmoid_228 => sigmoid_139
# Graph fragment:
#   %sub_127 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1017), kwargs = {})
#   %mul_521 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_127, %unsqueeze_1019), kwargs = {})
#   %mul_522 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_521, %unsqueeze_1021), kwargs = {})
#   %add_297 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_522, %unsqueeze_1023), kwargs = {})
#   %sigmoid_139 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_297,), kwargs = {})
#   %mul_523 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_297, %sigmoid_139), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2304)
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


# kernel path: inductor_cache/ty/ctyeagtdpz4glcveig6bf5rd4pnezgfi7ekv43kw3ay5yrh6ovlj.py
# Topologically Sorted Source Nodes: [features_49_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_49_conv_4 => add_299, mul_525, mul_526, sub_128
# Graph fragment:
#   %sub_128 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_1025), kwargs = {})
#   %mul_525 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_128, %unsqueeze_1027), kwargs = {})
#   %mul_526 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_525, %unsqueeze_1029), kwargs = {})
#   %add_299 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_526, %unsqueeze_1031), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_46', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_46', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2304)
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


# kernel path: inductor_cache/3o/c3o7hb6axtkraetuzvyfgmtrv36cxkekwabh473wtyj2meeefonc.py
# Topologically Sorted Source Nodes: [sigmoid_230, mul_141, features_49_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_49_conv_6_avg_pool => mean_30
#   mul_141 => mul_527
#   sigmoid_230 => sigmoid_140
# Graph fragment:
#   %sigmoid_140 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_299,), kwargs = {})
#   %mul_527 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_299, %sigmoid_140), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_527, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_47 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_47', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_47', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 2304)
    x1 = xindex // 2304
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 9216*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (2304 + x0 + 9216*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (4608 + x0 + 9216*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (6912 + x0 + 9216*x1), xmask)
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


# kernel path: inductor_cache/ei/ceipdhtg4vuswiyrnruonvp7y3trvsk6xlekqetz2uxgpdy3n3nz.py
# Topologically Sorted Source Nodes: [sigmoid_232, mul_142], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_142 => mul_528
#   sigmoid_232 => sigmoid_141
# Graph fragment:
#   %sigmoid_141 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_60,), kwargs = {})
#   %mul_528 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_60, %sigmoid_141), kwargs = {})
triton_poi_fused_mul_sigmoid_48 = async_compile.triton('triton_poi_fused_mul_sigmoid_48', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_48', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_48(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wl/cwlo7qmfeskj5upzk45g3jcbusnt5crrlkqdvak7e425w43qvspl.py
# Topologically Sorted Source Nodes: [sigmoid_230, mul_141, mul_143], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_141 => mul_527
#   mul_143 => mul_529
#   sigmoid_230 => sigmoid_140
# Graph fragment:
#   %sigmoid_140 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_299,), kwargs = {})
#   %mul_527 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_299, %sigmoid_140), kwargs = {})
#   %mul_529 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_527, %view_61), kwargs = {})
triton_poi_fused_mul_sigmoid_49 = async_compile.triton('triton_poi_fused_mul_sigmoid_49', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_49(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 2304)
    x2 = xindex // 9216
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 2304*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/xi/cxi2dah7ptqiowzl6zod33f3fckjaufcb2nozh3yzldwromgrb3q.py
# Topologically Sorted Source Nodes: [features_49_conv_8, add_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_43 => add_302
#   features_49_conv_8 => add_301, mul_531, mul_532, sub_129
# Graph fragment:
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_1033), kwargs = {})
#   %mul_531 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_1035), kwargs = {})
#   %mul_532 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_531, %unsqueeze_1037), kwargs = {})
#   %add_301 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_532, %unsqueeze_1039), kwargs = {})
#   %add_302 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_295, %add_301), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_50', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_50', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
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


# kernel path: inductor_cache/l7/cl7skb4cnsajeh355undz5m3jim6mo7k2l7esiogxwcq4od6ciot.py
# Topologically Sorted Source Nodes: [features_73_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_73_conv_8 => add_469, mul_843, mul_844, sub_201
# Graph fragment:
#   %sub_201 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_201, %unsqueeze_1609), kwargs = {})
#   %mul_843 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_201, %unsqueeze_1611), kwargs = {})
#   %mul_844 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_843, %unsqueeze_1613), kwargs = {})
#   %add_469 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_844, %unsqueeze_1615), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_51', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_51', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 640)
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


# kernel path: inductor_cache/f4/cf4b54b2c4wmahsqudltygk34pzrtn55pq6au5csoshwpbxwe5wi.py
# Topologically Sorted Source Nodes: [features_74_conv_1, sigmoid_403, mul_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_74_conv_1 => add_471, mul_846, mul_847, sub_202
#   mul_240 => mul_848
#   sigmoid_403 => sigmoid_239
# Graph fragment:
#   %sub_202 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_202, %unsqueeze_1617), kwargs = {})
#   %mul_846 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_202, %unsqueeze_1619), kwargs = {})
#   %mul_847 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_846, %unsqueeze_1621), kwargs = {})
#   %add_471 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_847, %unsqueeze_1623), kwargs = {})
#   %sigmoid_239 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_471,), kwargs = {})
#   %mul_848 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_471, %sigmoid_239), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3840)
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


# kernel path: inductor_cache/xb/cxbot6dvfwrzrbzunatwejg2oi2kv4jbzvi5sudakwiw3tjfixgq.py
# Topologically Sorted Source Nodes: [features_74_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_74_conv_4 => add_473, mul_850, mul_851, sub_203
# Graph fragment:
#   %sub_203 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_203, %unsqueeze_1625), kwargs = {})
#   %mul_850 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_203, %unsqueeze_1627), kwargs = {})
#   %mul_851 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_850, %unsqueeze_1629), kwargs = {})
#   %add_473 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_851, %unsqueeze_1631), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_53', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_53', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 3840)
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


# kernel path: inductor_cache/y6/cy6i47d5uswifm64grkbhei4obizsu6tiacjo6mymgr3ehg4ucwq.py
# Topologically Sorted Source Nodes: [sigmoid_405, mul_241, features_74_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_74_conv_6_avg_pool => mean_55
#   mul_241 => mul_852
#   sigmoid_405 => sigmoid_240
# Graph fragment:
#   %sigmoid_240 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_473,), kwargs = {})
#   %mul_852 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_473, %sigmoid_240), kwargs = {})
#   %mean_55 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_852, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_54 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_54', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_54', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_54(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3840)
    x1 = xindex // 3840
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 15360*x1), xmask)
    tmp3 = tl.load(in_ptr0 + (3840 + x0 + 15360*x1), xmask)
    tmp7 = tl.load(in_ptr0 + (7680 + x0 + 15360*x1), xmask)
    tmp11 = tl.load(in_ptr0 + (11520 + x0 + 15360*x1), xmask)
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


# kernel path: inductor_cache/fd/cfdiz7jd32sisomr6i4njtd6jqc7toe3z4pt2fba4y7xdu5zoeb2.py
# Topologically Sorted Source Nodes: [sigmoid_407, mul_242], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_242 => mul_853
#   sigmoid_407 => sigmoid_241
# Graph fragment:
#   %sigmoid_241 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_110,), kwargs = {})
#   %mul_853 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_110, %sigmoid_241), kwargs = {})
triton_poi_fused_mul_sigmoid_55 = async_compile.triton('triton_poi_fused_mul_sigmoid_55', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_55', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_55(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6r/c6rbq3exemsrvixb46mgn5ixlw2xnphxewvmivoqpzdbnsch7gwg.py
# Topologically Sorted Source Nodes: [sigmoid_405, mul_241, mul_243], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_241 => mul_852
#   mul_243 => mul_854
#   sigmoid_405 => sigmoid_240
# Graph fragment:
#   %sigmoid_240 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_473,), kwargs = {})
#   %mul_852 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_473, %sigmoid_240), kwargs = {})
#   %mul_854 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_852, %view_111), kwargs = {})
triton_poi_fused_mul_sigmoid_56 = async_compile.triton('triton_poi_fused_mul_sigmoid_56', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sigmoid_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sigmoid_56(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 61440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 3840)
    x2 = xindex // 15360
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 3840*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/ao/caoykxpw6kvg4j3gav6nfg3bq3jnqxngbzsisayxz7lyxva24mut.py
# Topologically Sorted Source Nodes: [features_74_conv_8, add_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_67 => add_476
#   features_74_conv_8 => add_475, mul_856, mul_857, sub_204
# Graph fragment:
#   %sub_204 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_204, %unsqueeze_1633), kwargs = {})
#   %mul_856 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_204, %unsqueeze_1635), kwargs = {})
#   %mul_857 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_856, %unsqueeze_1637), kwargs = {})
#   %add_475 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_857, %unsqueeze_1639), kwargs = {})
#   %add_476 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_469, %add_475), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_57', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_57', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 640)
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


# kernel path: inductor_cache/gz/cgz4hhhd3qtx3qu4jil6hkjmf4kmcvphmozdjfaowkwobemc6lai.py
# Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   conv_1 => add_513, mul_924, mul_925, sub_220
# Graph fragment:
#   %sub_220 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_220, %unsqueeze_1761), kwargs = {})
#   %mul_924 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_220, %unsqueeze_1763), kwargs = {})
#   %mul_925 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_924, %unsqueeze_1765), kwargs = {})
#   %add_513 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_925, %unsqueeze_1767), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_58', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_58', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/pn/cpnxwddrrfb7nqlhkmjocqw3mh2ysxqayfi6qsmjbz5tyau6oomv.py
# Topologically Sorted Source Nodes: [sigmoid_445, mul_264, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   avgpool => mean_61
#   mul_264 => mul_926
#   sigmoid_445 => sigmoid_263
# Graph fragment:
#   %sigmoid_263 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_513,), kwargs = {})
#   %mul_926 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_513, %sigmoid_263), kwargs = {})
#   %mean_61 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_926, [-1, -2], True), kwargs = {})
triton_poi_fused_mean_mul_sigmoid_59 = async_compile.triton('triton_poi_fused_mean_mul_sigmoid_59', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_mul_sigmoid_59', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_mul_sigmoid_59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352 = args
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
    assert_size_stride(primals_12, (32, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_13, (32, ), (1, ))
    assert_size_stride(primals_14, (32, ), (1, ))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, ), (1, ))
    assert_size_stride(primals_21, (32, ), (1, ))
    assert_size_stride(primals_22, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_23, (32, ), (1, ))
    assert_size_stride(primals_24, (32, ), (1, ))
    assert_size_stride(primals_25, (32, ), (1, ))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_27, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_28, (32, ), (1, ))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (32, ), (1, ))
    assert_size_stride(primals_32, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_33, (32, ), (1, ))
    assert_size_stride(primals_34, (32, ), (1, ))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_38, (32, ), (1, ))
    assert_size_stride(primals_39, (32, ), (1, ))
    assert_size_stride(primals_40, (32, ), (1, ))
    assert_size_stride(primals_41, (32, ), (1, ))
    assert_size_stride(primals_42, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_43, (32, ), (1, ))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, ), (1, ))
    assert_size_stride(primals_47, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (64, ), (1, ))
    assert_size_stride(primals_54, (64, ), (1, ))
    assert_size_stride(primals_55, (64, ), (1, ))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (64, ), (1, ))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (64, 256, 1, 1), (256, 1, 1, 1))
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
    assert_size_stride(primals_107, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_108, (256, ), (1, ))
    assert_size_stride(primals_109, (256, ), (1, ))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (64, ), (1, ))
    assert_size_stride(primals_117, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (256, ), (1, ))
    assert_size_stride(primals_121, (256, ), (1, ))
    assert_size_stride(primals_122, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_123, (96, ), (1, ))
    assert_size_stride(primals_124, (96, ), (1, ))
    assert_size_stride(primals_125, (96, ), (1, ))
    assert_size_stride(primals_126, (96, ), (1, ))
    assert_size_stride(primals_127, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_128, (384, ), (1, ))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (384, ), (1, ))
    assert_size_stride(primals_132, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_133, (96, ), (1, ))
    assert_size_stride(primals_134, (96, ), (1, ))
    assert_size_stride(primals_135, (96, ), (1, ))
    assert_size_stride(primals_136, (96, ), (1, ))
    assert_size_stride(primals_137, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, ), (1, ))
    assert_size_stride(primals_140, (384, ), (1, ))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_143, (96, ), (1, ))
    assert_size_stride(primals_144, (96, ), (1, ))
    assert_size_stride(primals_145, (96, ), (1, ))
    assert_size_stride(primals_146, (96, ), (1, ))
    assert_size_stride(primals_147, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_153, (96, ), (1, ))
    assert_size_stride(primals_154, (96, ), (1, ))
    assert_size_stride(primals_155, (96, ), (1, ))
    assert_size_stride(primals_156, (96, ), (1, ))
    assert_size_stride(primals_157, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_163, (96, ), (1, ))
    assert_size_stride(primals_164, (96, ), (1, ))
    assert_size_stride(primals_165, (96, ), (1, ))
    assert_size_stride(primals_166, (96, ), (1, ))
    assert_size_stride(primals_167, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_168, (384, ), (1, ))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_173, (96, ), (1, ))
    assert_size_stride(primals_174, (96, ), (1, ))
    assert_size_stride(primals_175, (96, ), (1, ))
    assert_size_stride(primals_176, (96, ), (1, ))
    assert_size_stride(primals_177, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (384, ), (1, ))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_182, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_183, (96, ), (1, ))
    assert_size_stride(primals_184, (96, ), (1, ))
    assert_size_stride(primals_185, (96, ), (1, ))
    assert_size_stride(primals_186, (96, ), (1, ))
    assert_size_stride(primals_187, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_188, (384, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (384, ), (1, ))
    assert_size_stride(primals_191, (384, ), (1, ))
    assert_size_stride(primals_192, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_197, (24, 384), (384, 1))
    assert_size_stride(primals_198, (24, ), (1, ))
    assert_size_stride(primals_199, (384, 24), (24, 1))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_202, (192, ), (1, ))
    assert_size_stride(primals_203, (192, ), (1, ))
    assert_size_stride(primals_204, (192, ), (1, ))
    assert_size_stride(primals_205, (192, ), (1, ))
    assert_size_stride(primals_206, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_207, (768, ), (1, ))
    assert_size_stride(primals_208, (768, ), (1, ))
    assert_size_stride(primals_209, (768, ), (1, ))
    assert_size_stride(primals_210, (768, ), (1, ))
    assert_size_stride(primals_211, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_212, (768, ), (1, ))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (48, 768), (768, 1))
    assert_size_stride(primals_217, (48, ), (1, ))
    assert_size_stride(primals_218, (768, 48), (48, 1))
    assert_size_stride(primals_219, (768, ), (1, ))
    assert_size_stride(primals_220, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_221, (192, ), (1, ))
    assert_size_stride(primals_222, (192, ), (1, ))
    assert_size_stride(primals_223, (192, ), (1, ))
    assert_size_stride(primals_224, (192, ), (1, ))
    assert_size_stride(primals_225, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_226, (768, ), (1, ))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (768, ), (1, ))
    assert_size_stride(primals_230, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_231, (768, ), (1, ))
    assert_size_stride(primals_232, (768, ), (1, ))
    assert_size_stride(primals_233, (768, ), (1, ))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (48, 768), (768, 1))
    assert_size_stride(primals_236, (48, ), (1, ))
    assert_size_stride(primals_237, (768, 48), (48, 1))
    assert_size_stride(primals_238, (768, ), (1, ))
    assert_size_stride(primals_239, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_240, (192, ), (1, ))
    assert_size_stride(primals_241, (192, ), (1, ))
    assert_size_stride(primals_242, (192, ), (1, ))
    assert_size_stride(primals_243, (192, ), (1, ))
    assert_size_stride(primals_244, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_245, (768, ), (1, ))
    assert_size_stride(primals_246, (768, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_250, (768, ), (1, ))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_254, (48, 768), (768, 1))
    assert_size_stride(primals_255, (48, ), (1, ))
    assert_size_stride(primals_256, (768, 48), (48, 1))
    assert_size_stride(primals_257, (768, ), (1, ))
    assert_size_stride(primals_258, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_259, (192, ), (1, ))
    assert_size_stride(primals_260, (192, ), (1, ))
    assert_size_stride(primals_261, (192, ), (1, ))
    assert_size_stride(primals_262, (192, ), (1, ))
    assert_size_stride(primals_263, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_264, (768, ), (1, ))
    assert_size_stride(primals_265, (768, ), (1, ))
    assert_size_stride(primals_266, (768, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_268, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_269, (768, ), (1, ))
    assert_size_stride(primals_270, (768, ), (1, ))
    assert_size_stride(primals_271, (768, ), (1, ))
    assert_size_stride(primals_272, (768, ), (1, ))
    assert_size_stride(primals_273, (48, 768), (768, 1))
    assert_size_stride(primals_274, (48, ), (1, ))
    assert_size_stride(primals_275, (768, 48), (48, 1))
    assert_size_stride(primals_276, (768, ), (1, ))
    assert_size_stride(primals_277, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_278, (192, ), (1, ))
    assert_size_stride(primals_279, (192, ), (1, ))
    assert_size_stride(primals_280, (192, ), (1, ))
    assert_size_stride(primals_281, (192, ), (1, ))
    assert_size_stride(primals_282, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_283, (768, ), (1, ))
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_285, (768, ), (1, ))
    assert_size_stride(primals_286, (768, ), (1, ))
    assert_size_stride(primals_287, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_288, (768, ), (1, ))
    assert_size_stride(primals_289, (768, ), (1, ))
    assert_size_stride(primals_290, (768, ), (1, ))
    assert_size_stride(primals_291, (768, ), (1, ))
    assert_size_stride(primals_292, (48, 768), (768, 1))
    assert_size_stride(primals_293, (48, ), (1, ))
    assert_size_stride(primals_294, (768, 48), (48, 1))
    assert_size_stride(primals_295, (768, ), (1, ))
    assert_size_stride(primals_296, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_297, (192, ), (1, ))
    assert_size_stride(primals_298, (192, ), (1, ))
    assert_size_stride(primals_299, (192, ), (1, ))
    assert_size_stride(primals_300, (192, ), (1, ))
    assert_size_stride(primals_301, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_302, (768, ), (1, ))
    assert_size_stride(primals_303, (768, ), (1, ))
    assert_size_stride(primals_304, (768, ), (1, ))
    assert_size_stride(primals_305, (768, ), (1, ))
    assert_size_stride(primals_306, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_307, (768, ), (1, ))
    assert_size_stride(primals_308, (768, ), (1, ))
    assert_size_stride(primals_309, (768, ), (1, ))
    assert_size_stride(primals_310, (768, ), (1, ))
    assert_size_stride(primals_311, (48, 768), (768, 1))
    assert_size_stride(primals_312, (48, ), (1, ))
    assert_size_stride(primals_313, (768, 48), (48, 1))
    assert_size_stride(primals_314, (768, ), (1, ))
    assert_size_stride(primals_315, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_316, (192, ), (1, ))
    assert_size_stride(primals_317, (192, ), (1, ))
    assert_size_stride(primals_318, (192, ), (1, ))
    assert_size_stride(primals_319, (192, ), (1, ))
    assert_size_stride(primals_320, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_321, (768, ), (1, ))
    assert_size_stride(primals_322, (768, ), (1, ))
    assert_size_stride(primals_323, (768, ), (1, ))
    assert_size_stride(primals_324, (768, ), (1, ))
    assert_size_stride(primals_325, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_326, (768, ), (1, ))
    assert_size_stride(primals_327, (768, ), (1, ))
    assert_size_stride(primals_328, (768, ), (1, ))
    assert_size_stride(primals_329, (768, ), (1, ))
    assert_size_stride(primals_330, (48, 768), (768, 1))
    assert_size_stride(primals_331, (48, ), (1, ))
    assert_size_stride(primals_332, (768, 48), (48, 1))
    assert_size_stride(primals_333, (768, ), (1, ))
    assert_size_stride(primals_334, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_335, (192, ), (1, ))
    assert_size_stride(primals_336, (192, ), (1, ))
    assert_size_stride(primals_337, (192, ), (1, ))
    assert_size_stride(primals_338, (192, ), (1, ))
    assert_size_stride(primals_339, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_340, (768, ), (1, ))
    assert_size_stride(primals_341, (768, ), (1, ))
    assert_size_stride(primals_342, (768, ), (1, ))
    assert_size_stride(primals_343, (768, ), (1, ))
    assert_size_stride(primals_344, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_345, (768, ), (1, ))
    assert_size_stride(primals_346, (768, ), (1, ))
    assert_size_stride(primals_347, (768, ), (1, ))
    assert_size_stride(primals_348, (768, ), (1, ))
    assert_size_stride(primals_349, (48, 768), (768, 1))
    assert_size_stride(primals_350, (48, ), (1, ))
    assert_size_stride(primals_351, (768, 48), (48, 1))
    assert_size_stride(primals_352, (768, ), (1, ))
    assert_size_stride(primals_353, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_354, (192, ), (1, ))
    assert_size_stride(primals_355, (192, ), (1, ))
    assert_size_stride(primals_356, (192, ), (1, ))
    assert_size_stride(primals_357, (192, ), (1, ))
    assert_size_stride(primals_358, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_359, (768, ), (1, ))
    assert_size_stride(primals_360, (768, ), (1, ))
    assert_size_stride(primals_361, (768, ), (1, ))
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_364, (768, ), (1, ))
    assert_size_stride(primals_365, (768, ), (1, ))
    assert_size_stride(primals_366, (768, ), (1, ))
    assert_size_stride(primals_367, (768, ), (1, ))
    assert_size_stride(primals_368, (48, 768), (768, 1))
    assert_size_stride(primals_369, (48, ), (1, ))
    assert_size_stride(primals_370, (768, 48), (48, 1))
    assert_size_stride(primals_371, (768, ), (1, ))
    assert_size_stride(primals_372, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_373, (192, ), (1, ))
    assert_size_stride(primals_374, (192, ), (1, ))
    assert_size_stride(primals_375, (192, ), (1, ))
    assert_size_stride(primals_376, (192, ), (1, ))
    assert_size_stride(primals_377, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_378, (1152, ), (1, ))
    assert_size_stride(primals_379, (1152, ), (1, ))
    assert_size_stride(primals_380, (1152, ), (1, ))
    assert_size_stride(primals_381, (1152, ), (1, ))
    assert_size_stride(primals_382, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_383, (1152, ), (1, ))
    assert_size_stride(primals_384, (1152, ), (1, ))
    assert_size_stride(primals_385, (1152, ), (1, ))
    assert_size_stride(primals_386, (1152, ), (1, ))
    assert_size_stride(primals_387, (48, 1152), (1152, 1))
    assert_size_stride(primals_388, (48, ), (1, ))
    assert_size_stride(primals_389, (1152, 48), (48, 1))
    assert_size_stride(primals_390, (1152, ), (1, ))
    assert_size_stride(primals_391, (224, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_392, (224, ), (1, ))
    assert_size_stride(primals_393, (224, ), (1, ))
    assert_size_stride(primals_394, (224, ), (1, ))
    assert_size_stride(primals_395, (224, ), (1, ))
    assert_size_stride(primals_396, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_397, (1344, ), (1, ))
    assert_size_stride(primals_398, (1344, ), (1, ))
    assert_size_stride(primals_399, (1344, ), (1, ))
    assert_size_stride(primals_400, (1344, ), (1, ))
    assert_size_stride(primals_401, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_402, (1344, ), (1, ))
    assert_size_stride(primals_403, (1344, ), (1, ))
    assert_size_stride(primals_404, (1344, ), (1, ))
    assert_size_stride(primals_405, (1344, ), (1, ))
    assert_size_stride(primals_406, (56, 1344), (1344, 1))
    assert_size_stride(primals_407, (56, ), (1, ))
    assert_size_stride(primals_408, (1344, 56), (56, 1))
    assert_size_stride(primals_409, (1344, ), (1, ))
    assert_size_stride(primals_410, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_411, (224, ), (1, ))
    assert_size_stride(primals_412, (224, ), (1, ))
    assert_size_stride(primals_413, (224, ), (1, ))
    assert_size_stride(primals_414, (224, ), (1, ))
    assert_size_stride(primals_415, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_416, (1344, ), (1, ))
    assert_size_stride(primals_417, (1344, ), (1, ))
    assert_size_stride(primals_418, (1344, ), (1, ))
    assert_size_stride(primals_419, (1344, ), (1, ))
    assert_size_stride(primals_420, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_421, (1344, ), (1, ))
    assert_size_stride(primals_422, (1344, ), (1, ))
    assert_size_stride(primals_423, (1344, ), (1, ))
    assert_size_stride(primals_424, (1344, ), (1, ))
    assert_size_stride(primals_425, (56, 1344), (1344, 1))
    assert_size_stride(primals_426, (56, ), (1, ))
    assert_size_stride(primals_427, (1344, 56), (56, 1))
    assert_size_stride(primals_428, (1344, ), (1, ))
    assert_size_stride(primals_429, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_430, (224, ), (1, ))
    assert_size_stride(primals_431, (224, ), (1, ))
    assert_size_stride(primals_432, (224, ), (1, ))
    assert_size_stride(primals_433, (224, ), (1, ))
    assert_size_stride(primals_434, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_435, (1344, ), (1, ))
    assert_size_stride(primals_436, (1344, ), (1, ))
    assert_size_stride(primals_437, (1344, ), (1, ))
    assert_size_stride(primals_438, (1344, ), (1, ))
    assert_size_stride(primals_439, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_440, (1344, ), (1, ))
    assert_size_stride(primals_441, (1344, ), (1, ))
    assert_size_stride(primals_442, (1344, ), (1, ))
    assert_size_stride(primals_443, (1344, ), (1, ))
    assert_size_stride(primals_444, (56, 1344), (1344, 1))
    assert_size_stride(primals_445, (56, ), (1, ))
    assert_size_stride(primals_446, (1344, 56), (56, 1))
    assert_size_stride(primals_447, (1344, ), (1, ))
    assert_size_stride(primals_448, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_449, (224, ), (1, ))
    assert_size_stride(primals_450, (224, ), (1, ))
    assert_size_stride(primals_451, (224, ), (1, ))
    assert_size_stride(primals_452, (224, ), (1, ))
    assert_size_stride(primals_453, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_454, (1344, ), (1, ))
    assert_size_stride(primals_455, (1344, ), (1, ))
    assert_size_stride(primals_456, (1344, ), (1, ))
    assert_size_stride(primals_457, (1344, ), (1, ))
    assert_size_stride(primals_458, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_459, (1344, ), (1, ))
    assert_size_stride(primals_460, (1344, ), (1, ))
    assert_size_stride(primals_461, (1344, ), (1, ))
    assert_size_stride(primals_462, (1344, ), (1, ))
    assert_size_stride(primals_463, (56, 1344), (1344, 1))
    assert_size_stride(primals_464, (56, ), (1, ))
    assert_size_stride(primals_465, (1344, 56), (56, 1))
    assert_size_stride(primals_466, (1344, ), (1, ))
    assert_size_stride(primals_467, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_468, (224, ), (1, ))
    assert_size_stride(primals_469, (224, ), (1, ))
    assert_size_stride(primals_470, (224, ), (1, ))
    assert_size_stride(primals_471, (224, ), (1, ))
    assert_size_stride(primals_472, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_473, (1344, ), (1, ))
    assert_size_stride(primals_474, (1344, ), (1, ))
    assert_size_stride(primals_475, (1344, ), (1, ))
    assert_size_stride(primals_476, (1344, ), (1, ))
    assert_size_stride(primals_477, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_478, (1344, ), (1, ))
    assert_size_stride(primals_479, (1344, ), (1, ))
    assert_size_stride(primals_480, (1344, ), (1, ))
    assert_size_stride(primals_481, (1344, ), (1, ))
    assert_size_stride(primals_482, (56, 1344), (1344, 1))
    assert_size_stride(primals_483, (56, ), (1, ))
    assert_size_stride(primals_484, (1344, 56), (56, 1))
    assert_size_stride(primals_485, (1344, ), (1, ))
    assert_size_stride(primals_486, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_487, (224, ), (1, ))
    assert_size_stride(primals_488, (224, ), (1, ))
    assert_size_stride(primals_489, (224, ), (1, ))
    assert_size_stride(primals_490, (224, ), (1, ))
    assert_size_stride(primals_491, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_492, (1344, ), (1, ))
    assert_size_stride(primals_493, (1344, ), (1, ))
    assert_size_stride(primals_494, (1344, ), (1, ))
    assert_size_stride(primals_495, (1344, ), (1, ))
    assert_size_stride(primals_496, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_497, (1344, ), (1, ))
    assert_size_stride(primals_498, (1344, ), (1, ))
    assert_size_stride(primals_499, (1344, ), (1, ))
    assert_size_stride(primals_500, (1344, ), (1, ))
    assert_size_stride(primals_501, (56, 1344), (1344, 1))
    assert_size_stride(primals_502, (56, ), (1, ))
    assert_size_stride(primals_503, (1344, 56), (56, 1))
    assert_size_stride(primals_504, (1344, ), (1, ))
    assert_size_stride(primals_505, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_506, (224, ), (1, ))
    assert_size_stride(primals_507, (224, ), (1, ))
    assert_size_stride(primals_508, (224, ), (1, ))
    assert_size_stride(primals_509, (224, ), (1, ))
    assert_size_stride(primals_510, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_511, (1344, ), (1, ))
    assert_size_stride(primals_512, (1344, ), (1, ))
    assert_size_stride(primals_513, (1344, ), (1, ))
    assert_size_stride(primals_514, (1344, ), (1, ))
    assert_size_stride(primals_515, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_516, (1344, ), (1, ))
    assert_size_stride(primals_517, (1344, ), (1, ))
    assert_size_stride(primals_518, (1344, ), (1, ))
    assert_size_stride(primals_519, (1344, ), (1, ))
    assert_size_stride(primals_520, (56, 1344), (1344, 1))
    assert_size_stride(primals_521, (56, ), (1, ))
    assert_size_stride(primals_522, (1344, 56), (56, 1))
    assert_size_stride(primals_523, (1344, ), (1, ))
    assert_size_stride(primals_524, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_525, (224, ), (1, ))
    assert_size_stride(primals_526, (224, ), (1, ))
    assert_size_stride(primals_527, (224, ), (1, ))
    assert_size_stride(primals_528, (224, ), (1, ))
    assert_size_stride(primals_529, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_530, (1344, ), (1, ))
    assert_size_stride(primals_531, (1344, ), (1, ))
    assert_size_stride(primals_532, (1344, ), (1, ))
    assert_size_stride(primals_533, (1344, ), (1, ))
    assert_size_stride(primals_534, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_535, (1344, ), (1, ))
    assert_size_stride(primals_536, (1344, ), (1, ))
    assert_size_stride(primals_537, (1344, ), (1, ))
    assert_size_stride(primals_538, (1344, ), (1, ))
    assert_size_stride(primals_539, (56, 1344), (1344, 1))
    assert_size_stride(primals_540, (56, ), (1, ))
    assert_size_stride(primals_541, (1344, 56), (56, 1))
    assert_size_stride(primals_542, (1344, ), (1, ))
    assert_size_stride(primals_543, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_544, (224, ), (1, ))
    assert_size_stride(primals_545, (224, ), (1, ))
    assert_size_stride(primals_546, (224, ), (1, ))
    assert_size_stride(primals_547, (224, ), (1, ))
    assert_size_stride(primals_548, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_549, (1344, ), (1, ))
    assert_size_stride(primals_550, (1344, ), (1, ))
    assert_size_stride(primals_551, (1344, ), (1, ))
    assert_size_stride(primals_552, (1344, ), (1, ))
    assert_size_stride(primals_553, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_554, (1344, ), (1, ))
    assert_size_stride(primals_555, (1344, ), (1, ))
    assert_size_stride(primals_556, (1344, ), (1, ))
    assert_size_stride(primals_557, (1344, ), (1, ))
    assert_size_stride(primals_558, (56, 1344), (1344, 1))
    assert_size_stride(primals_559, (56, ), (1, ))
    assert_size_stride(primals_560, (1344, 56), (56, 1))
    assert_size_stride(primals_561, (1344, ), (1, ))
    assert_size_stride(primals_562, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_563, (224, ), (1, ))
    assert_size_stride(primals_564, (224, ), (1, ))
    assert_size_stride(primals_565, (224, ), (1, ))
    assert_size_stride(primals_566, (224, ), (1, ))
    assert_size_stride(primals_567, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_568, (1344, ), (1, ))
    assert_size_stride(primals_569, (1344, ), (1, ))
    assert_size_stride(primals_570, (1344, ), (1, ))
    assert_size_stride(primals_571, (1344, ), (1, ))
    assert_size_stride(primals_572, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_573, (1344, ), (1, ))
    assert_size_stride(primals_574, (1344, ), (1, ))
    assert_size_stride(primals_575, (1344, ), (1, ))
    assert_size_stride(primals_576, (1344, ), (1, ))
    assert_size_stride(primals_577, (56, 1344), (1344, 1))
    assert_size_stride(primals_578, (56, ), (1, ))
    assert_size_stride(primals_579, (1344, 56), (56, 1))
    assert_size_stride(primals_580, (1344, ), (1, ))
    assert_size_stride(primals_581, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_582, (224, ), (1, ))
    assert_size_stride(primals_583, (224, ), (1, ))
    assert_size_stride(primals_584, (224, ), (1, ))
    assert_size_stride(primals_585, (224, ), (1, ))
    assert_size_stride(primals_586, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_587, (1344, ), (1, ))
    assert_size_stride(primals_588, (1344, ), (1, ))
    assert_size_stride(primals_589, (1344, ), (1, ))
    assert_size_stride(primals_590, (1344, ), (1, ))
    assert_size_stride(primals_591, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_592, (1344, ), (1, ))
    assert_size_stride(primals_593, (1344, ), (1, ))
    assert_size_stride(primals_594, (1344, ), (1, ))
    assert_size_stride(primals_595, (1344, ), (1, ))
    assert_size_stride(primals_596, (56, 1344), (1344, 1))
    assert_size_stride(primals_597, (56, ), (1, ))
    assert_size_stride(primals_598, (1344, 56), (56, 1))
    assert_size_stride(primals_599, (1344, ), (1, ))
    assert_size_stride(primals_600, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_601, (224, ), (1, ))
    assert_size_stride(primals_602, (224, ), (1, ))
    assert_size_stride(primals_603, (224, ), (1, ))
    assert_size_stride(primals_604, (224, ), (1, ))
    assert_size_stride(primals_605, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_606, (1344, ), (1, ))
    assert_size_stride(primals_607, (1344, ), (1, ))
    assert_size_stride(primals_608, (1344, ), (1, ))
    assert_size_stride(primals_609, (1344, ), (1, ))
    assert_size_stride(primals_610, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_611, (1344, ), (1, ))
    assert_size_stride(primals_612, (1344, ), (1, ))
    assert_size_stride(primals_613, (1344, ), (1, ))
    assert_size_stride(primals_614, (1344, ), (1, ))
    assert_size_stride(primals_615, (56, 1344), (1344, 1))
    assert_size_stride(primals_616, (56, ), (1, ))
    assert_size_stride(primals_617, (1344, 56), (56, 1))
    assert_size_stride(primals_618, (1344, ), (1, ))
    assert_size_stride(primals_619, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_620, (224, ), (1, ))
    assert_size_stride(primals_621, (224, ), (1, ))
    assert_size_stride(primals_622, (224, ), (1, ))
    assert_size_stride(primals_623, (224, ), (1, ))
    assert_size_stride(primals_624, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_625, (1344, ), (1, ))
    assert_size_stride(primals_626, (1344, ), (1, ))
    assert_size_stride(primals_627, (1344, ), (1, ))
    assert_size_stride(primals_628, (1344, ), (1, ))
    assert_size_stride(primals_629, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_630, (1344, ), (1, ))
    assert_size_stride(primals_631, (1344, ), (1, ))
    assert_size_stride(primals_632, (1344, ), (1, ))
    assert_size_stride(primals_633, (1344, ), (1, ))
    assert_size_stride(primals_634, (56, 1344), (1344, 1))
    assert_size_stride(primals_635, (56, ), (1, ))
    assert_size_stride(primals_636, (1344, 56), (56, 1))
    assert_size_stride(primals_637, (1344, ), (1, ))
    assert_size_stride(primals_638, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_639, (224, ), (1, ))
    assert_size_stride(primals_640, (224, ), (1, ))
    assert_size_stride(primals_641, (224, ), (1, ))
    assert_size_stride(primals_642, (224, ), (1, ))
    assert_size_stride(primals_643, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_644, (1344, ), (1, ))
    assert_size_stride(primals_645, (1344, ), (1, ))
    assert_size_stride(primals_646, (1344, ), (1, ))
    assert_size_stride(primals_647, (1344, ), (1, ))
    assert_size_stride(primals_648, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_649, (1344, ), (1, ))
    assert_size_stride(primals_650, (1344, ), (1, ))
    assert_size_stride(primals_651, (1344, ), (1, ))
    assert_size_stride(primals_652, (1344, ), (1, ))
    assert_size_stride(primals_653, (56, 1344), (1344, 1))
    assert_size_stride(primals_654, (56, ), (1, ))
    assert_size_stride(primals_655, (1344, 56), (56, 1))
    assert_size_stride(primals_656, (1344, ), (1, ))
    assert_size_stride(primals_657, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_658, (224, ), (1, ))
    assert_size_stride(primals_659, (224, ), (1, ))
    assert_size_stride(primals_660, (224, ), (1, ))
    assert_size_stride(primals_661, (224, ), (1, ))
    assert_size_stride(primals_662, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_663, (1344, ), (1, ))
    assert_size_stride(primals_664, (1344, ), (1, ))
    assert_size_stride(primals_665, (1344, ), (1, ))
    assert_size_stride(primals_666, (1344, ), (1, ))
    assert_size_stride(primals_667, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_668, (1344, ), (1, ))
    assert_size_stride(primals_669, (1344, ), (1, ))
    assert_size_stride(primals_670, (1344, ), (1, ))
    assert_size_stride(primals_671, (1344, ), (1, ))
    assert_size_stride(primals_672, (56, 1344), (1344, 1))
    assert_size_stride(primals_673, (56, ), (1, ))
    assert_size_stride(primals_674, (1344, 56), (56, 1))
    assert_size_stride(primals_675, (1344, ), (1, ))
    assert_size_stride(primals_676, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_677, (224, ), (1, ))
    assert_size_stride(primals_678, (224, ), (1, ))
    assert_size_stride(primals_679, (224, ), (1, ))
    assert_size_stride(primals_680, (224, ), (1, ))
    assert_size_stride(primals_681, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_682, (1344, ), (1, ))
    assert_size_stride(primals_683, (1344, ), (1, ))
    assert_size_stride(primals_684, (1344, ), (1, ))
    assert_size_stride(primals_685, (1344, ), (1, ))
    assert_size_stride(primals_686, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_687, (1344, ), (1, ))
    assert_size_stride(primals_688, (1344, ), (1, ))
    assert_size_stride(primals_689, (1344, ), (1, ))
    assert_size_stride(primals_690, (1344, ), (1, ))
    assert_size_stride(primals_691, (56, 1344), (1344, 1))
    assert_size_stride(primals_692, (56, ), (1, ))
    assert_size_stride(primals_693, (1344, 56), (56, 1))
    assert_size_stride(primals_694, (1344, ), (1, ))
    assert_size_stride(primals_695, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_696, (224, ), (1, ))
    assert_size_stride(primals_697, (224, ), (1, ))
    assert_size_stride(primals_698, (224, ), (1, ))
    assert_size_stride(primals_699, (224, ), (1, ))
    assert_size_stride(primals_700, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_701, (1344, ), (1, ))
    assert_size_stride(primals_702, (1344, ), (1, ))
    assert_size_stride(primals_703, (1344, ), (1, ))
    assert_size_stride(primals_704, (1344, ), (1, ))
    assert_size_stride(primals_705, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_706, (1344, ), (1, ))
    assert_size_stride(primals_707, (1344, ), (1, ))
    assert_size_stride(primals_708, (1344, ), (1, ))
    assert_size_stride(primals_709, (1344, ), (1, ))
    assert_size_stride(primals_710, (56, 1344), (1344, 1))
    assert_size_stride(primals_711, (56, ), (1, ))
    assert_size_stride(primals_712, (1344, 56), (56, 1))
    assert_size_stride(primals_713, (1344, ), (1, ))
    assert_size_stride(primals_714, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_715, (224, ), (1, ))
    assert_size_stride(primals_716, (224, ), (1, ))
    assert_size_stride(primals_717, (224, ), (1, ))
    assert_size_stride(primals_718, (224, ), (1, ))
    assert_size_stride(primals_719, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_720, (1344, ), (1, ))
    assert_size_stride(primals_721, (1344, ), (1, ))
    assert_size_stride(primals_722, (1344, ), (1, ))
    assert_size_stride(primals_723, (1344, ), (1, ))
    assert_size_stride(primals_724, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_725, (1344, ), (1, ))
    assert_size_stride(primals_726, (1344, ), (1, ))
    assert_size_stride(primals_727, (1344, ), (1, ))
    assert_size_stride(primals_728, (1344, ), (1, ))
    assert_size_stride(primals_729, (56, 1344), (1344, 1))
    assert_size_stride(primals_730, (56, ), (1, ))
    assert_size_stride(primals_731, (1344, 56), (56, 1))
    assert_size_stride(primals_732, (1344, ), (1, ))
    assert_size_stride(primals_733, (224, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_734, (224, ), (1, ))
    assert_size_stride(primals_735, (224, ), (1, ))
    assert_size_stride(primals_736, (224, ), (1, ))
    assert_size_stride(primals_737, (224, ), (1, ))
    assert_size_stride(primals_738, (1344, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_739, (1344, ), (1, ))
    assert_size_stride(primals_740, (1344, ), (1, ))
    assert_size_stride(primals_741, (1344, ), (1, ))
    assert_size_stride(primals_742, (1344, ), (1, ))
    assert_size_stride(primals_743, (1344, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_744, (1344, ), (1, ))
    assert_size_stride(primals_745, (1344, ), (1, ))
    assert_size_stride(primals_746, (1344, ), (1, ))
    assert_size_stride(primals_747, (1344, ), (1, ))
    assert_size_stride(primals_748, (56, 1344), (1344, 1))
    assert_size_stride(primals_749, (56, ), (1, ))
    assert_size_stride(primals_750, (1344, 56), (56, 1))
    assert_size_stride(primals_751, (1344, ), (1, ))
    assert_size_stride(primals_752, (384, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_753, (384, ), (1, ))
    assert_size_stride(primals_754, (384, ), (1, ))
    assert_size_stride(primals_755, (384, ), (1, ))
    assert_size_stride(primals_756, (384, ), (1, ))
    assert_size_stride(primals_757, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_758, (2304, ), (1, ))
    assert_size_stride(primals_759, (2304, ), (1, ))
    assert_size_stride(primals_760, (2304, ), (1, ))
    assert_size_stride(primals_761, (2304, ), (1, ))
    assert_size_stride(primals_762, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_763, (2304, ), (1, ))
    assert_size_stride(primals_764, (2304, ), (1, ))
    assert_size_stride(primals_765, (2304, ), (1, ))
    assert_size_stride(primals_766, (2304, ), (1, ))
    assert_size_stride(primals_767, (96, 2304), (2304, 1))
    assert_size_stride(primals_768, (96, ), (1, ))
    assert_size_stride(primals_769, (2304, 96), (96, 1))
    assert_size_stride(primals_770, (2304, ), (1, ))
    assert_size_stride(primals_771, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_772, (384, ), (1, ))
    assert_size_stride(primals_773, (384, ), (1, ))
    assert_size_stride(primals_774, (384, ), (1, ))
    assert_size_stride(primals_775, (384, ), (1, ))
    assert_size_stride(primals_776, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_777, (2304, ), (1, ))
    assert_size_stride(primals_778, (2304, ), (1, ))
    assert_size_stride(primals_779, (2304, ), (1, ))
    assert_size_stride(primals_780, (2304, ), (1, ))
    assert_size_stride(primals_781, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_782, (2304, ), (1, ))
    assert_size_stride(primals_783, (2304, ), (1, ))
    assert_size_stride(primals_784, (2304, ), (1, ))
    assert_size_stride(primals_785, (2304, ), (1, ))
    assert_size_stride(primals_786, (96, 2304), (2304, 1))
    assert_size_stride(primals_787, (96, ), (1, ))
    assert_size_stride(primals_788, (2304, 96), (96, 1))
    assert_size_stride(primals_789, (2304, ), (1, ))
    assert_size_stride(primals_790, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_791, (384, ), (1, ))
    assert_size_stride(primals_792, (384, ), (1, ))
    assert_size_stride(primals_793, (384, ), (1, ))
    assert_size_stride(primals_794, (384, ), (1, ))
    assert_size_stride(primals_795, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_796, (2304, ), (1, ))
    assert_size_stride(primals_797, (2304, ), (1, ))
    assert_size_stride(primals_798, (2304, ), (1, ))
    assert_size_stride(primals_799, (2304, ), (1, ))
    assert_size_stride(primals_800, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_801, (2304, ), (1, ))
    assert_size_stride(primals_802, (2304, ), (1, ))
    assert_size_stride(primals_803, (2304, ), (1, ))
    assert_size_stride(primals_804, (2304, ), (1, ))
    assert_size_stride(primals_805, (96, 2304), (2304, 1))
    assert_size_stride(primals_806, (96, ), (1, ))
    assert_size_stride(primals_807, (2304, 96), (96, 1))
    assert_size_stride(primals_808, (2304, ), (1, ))
    assert_size_stride(primals_809, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_810, (384, ), (1, ))
    assert_size_stride(primals_811, (384, ), (1, ))
    assert_size_stride(primals_812, (384, ), (1, ))
    assert_size_stride(primals_813, (384, ), (1, ))
    assert_size_stride(primals_814, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_815, (2304, ), (1, ))
    assert_size_stride(primals_816, (2304, ), (1, ))
    assert_size_stride(primals_817, (2304, ), (1, ))
    assert_size_stride(primals_818, (2304, ), (1, ))
    assert_size_stride(primals_819, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_820, (2304, ), (1, ))
    assert_size_stride(primals_821, (2304, ), (1, ))
    assert_size_stride(primals_822, (2304, ), (1, ))
    assert_size_stride(primals_823, (2304, ), (1, ))
    assert_size_stride(primals_824, (96, 2304), (2304, 1))
    assert_size_stride(primals_825, (96, ), (1, ))
    assert_size_stride(primals_826, (2304, 96), (96, 1))
    assert_size_stride(primals_827, (2304, ), (1, ))
    assert_size_stride(primals_828, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_829, (384, ), (1, ))
    assert_size_stride(primals_830, (384, ), (1, ))
    assert_size_stride(primals_831, (384, ), (1, ))
    assert_size_stride(primals_832, (384, ), (1, ))
    assert_size_stride(primals_833, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_834, (2304, ), (1, ))
    assert_size_stride(primals_835, (2304, ), (1, ))
    assert_size_stride(primals_836, (2304, ), (1, ))
    assert_size_stride(primals_837, (2304, ), (1, ))
    assert_size_stride(primals_838, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_839, (2304, ), (1, ))
    assert_size_stride(primals_840, (2304, ), (1, ))
    assert_size_stride(primals_841, (2304, ), (1, ))
    assert_size_stride(primals_842, (2304, ), (1, ))
    assert_size_stride(primals_843, (96, 2304), (2304, 1))
    assert_size_stride(primals_844, (96, ), (1, ))
    assert_size_stride(primals_845, (2304, 96), (96, 1))
    assert_size_stride(primals_846, (2304, ), (1, ))
    assert_size_stride(primals_847, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_848, (384, ), (1, ))
    assert_size_stride(primals_849, (384, ), (1, ))
    assert_size_stride(primals_850, (384, ), (1, ))
    assert_size_stride(primals_851, (384, ), (1, ))
    assert_size_stride(primals_852, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_853, (2304, ), (1, ))
    assert_size_stride(primals_854, (2304, ), (1, ))
    assert_size_stride(primals_855, (2304, ), (1, ))
    assert_size_stride(primals_856, (2304, ), (1, ))
    assert_size_stride(primals_857, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_858, (2304, ), (1, ))
    assert_size_stride(primals_859, (2304, ), (1, ))
    assert_size_stride(primals_860, (2304, ), (1, ))
    assert_size_stride(primals_861, (2304, ), (1, ))
    assert_size_stride(primals_862, (96, 2304), (2304, 1))
    assert_size_stride(primals_863, (96, ), (1, ))
    assert_size_stride(primals_864, (2304, 96), (96, 1))
    assert_size_stride(primals_865, (2304, ), (1, ))
    assert_size_stride(primals_866, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_867, (384, ), (1, ))
    assert_size_stride(primals_868, (384, ), (1, ))
    assert_size_stride(primals_869, (384, ), (1, ))
    assert_size_stride(primals_870, (384, ), (1, ))
    assert_size_stride(primals_871, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_872, (2304, ), (1, ))
    assert_size_stride(primals_873, (2304, ), (1, ))
    assert_size_stride(primals_874, (2304, ), (1, ))
    assert_size_stride(primals_875, (2304, ), (1, ))
    assert_size_stride(primals_876, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_877, (2304, ), (1, ))
    assert_size_stride(primals_878, (2304, ), (1, ))
    assert_size_stride(primals_879, (2304, ), (1, ))
    assert_size_stride(primals_880, (2304, ), (1, ))
    assert_size_stride(primals_881, (96, 2304), (2304, 1))
    assert_size_stride(primals_882, (96, ), (1, ))
    assert_size_stride(primals_883, (2304, 96), (96, 1))
    assert_size_stride(primals_884, (2304, ), (1, ))
    assert_size_stride(primals_885, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_886, (384, ), (1, ))
    assert_size_stride(primals_887, (384, ), (1, ))
    assert_size_stride(primals_888, (384, ), (1, ))
    assert_size_stride(primals_889, (384, ), (1, ))
    assert_size_stride(primals_890, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_891, (2304, ), (1, ))
    assert_size_stride(primals_892, (2304, ), (1, ))
    assert_size_stride(primals_893, (2304, ), (1, ))
    assert_size_stride(primals_894, (2304, ), (1, ))
    assert_size_stride(primals_895, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_896, (2304, ), (1, ))
    assert_size_stride(primals_897, (2304, ), (1, ))
    assert_size_stride(primals_898, (2304, ), (1, ))
    assert_size_stride(primals_899, (2304, ), (1, ))
    assert_size_stride(primals_900, (96, 2304), (2304, 1))
    assert_size_stride(primals_901, (96, ), (1, ))
    assert_size_stride(primals_902, (2304, 96), (96, 1))
    assert_size_stride(primals_903, (2304, ), (1, ))
    assert_size_stride(primals_904, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_905, (384, ), (1, ))
    assert_size_stride(primals_906, (384, ), (1, ))
    assert_size_stride(primals_907, (384, ), (1, ))
    assert_size_stride(primals_908, (384, ), (1, ))
    assert_size_stride(primals_909, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_910, (2304, ), (1, ))
    assert_size_stride(primals_911, (2304, ), (1, ))
    assert_size_stride(primals_912, (2304, ), (1, ))
    assert_size_stride(primals_913, (2304, ), (1, ))
    assert_size_stride(primals_914, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_915, (2304, ), (1, ))
    assert_size_stride(primals_916, (2304, ), (1, ))
    assert_size_stride(primals_917, (2304, ), (1, ))
    assert_size_stride(primals_918, (2304, ), (1, ))
    assert_size_stride(primals_919, (96, 2304), (2304, 1))
    assert_size_stride(primals_920, (96, ), (1, ))
    assert_size_stride(primals_921, (2304, 96), (96, 1))
    assert_size_stride(primals_922, (2304, ), (1, ))
    assert_size_stride(primals_923, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_924, (384, ), (1, ))
    assert_size_stride(primals_925, (384, ), (1, ))
    assert_size_stride(primals_926, (384, ), (1, ))
    assert_size_stride(primals_927, (384, ), (1, ))
    assert_size_stride(primals_928, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_929, (2304, ), (1, ))
    assert_size_stride(primals_930, (2304, ), (1, ))
    assert_size_stride(primals_931, (2304, ), (1, ))
    assert_size_stride(primals_932, (2304, ), (1, ))
    assert_size_stride(primals_933, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_934, (2304, ), (1, ))
    assert_size_stride(primals_935, (2304, ), (1, ))
    assert_size_stride(primals_936, (2304, ), (1, ))
    assert_size_stride(primals_937, (2304, ), (1, ))
    assert_size_stride(primals_938, (96, 2304), (2304, 1))
    assert_size_stride(primals_939, (96, ), (1, ))
    assert_size_stride(primals_940, (2304, 96), (96, 1))
    assert_size_stride(primals_941, (2304, ), (1, ))
    assert_size_stride(primals_942, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_943, (384, ), (1, ))
    assert_size_stride(primals_944, (384, ), (1, ))
    assert_size_stride(primals_945, (384, ), (1, ))
    assert_size_stride(primals_946, (384, ), (1, ))
    assert_size_stride(primals_947, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_948, (2304, ), (1, ))
    assert_size_stride(primals_949, (2304, ), (1, ))
    assert_size_stride(primals_950, (2304, ), (1, ))
    assert_size_stride(primals_951, (2304, ), (1, ))
    assert_size_stride(primals_952, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_953, (2304, ), (1, ))
    assert_size_stride(primals_954, (2304, ), (1, ))
    assert_size_stride(primals_955, (2304, ), (1, ))
    assert_size_stride(primals_956, (2304, ), (1, ))
    assert_size_stride(primals_957, (96, 2304), (2304, 1))
    assert_size_stride(primals_958, (96, ), (1, ))
    assert_size_stride(primals_959, (2304, 96), (96, 1))
    assert_size_stride(primals_960, (2304, ), (1, ))
    assert_size_stride(primals_961, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_962, (384, ), (1, ))
    assert_size_stride(primals_963, (384, ), (1, ))
    assert_size_stride(primals_964, (384, ), (1, ))
    assert_size_stride(primals_965, (384, ), (1, ))
    assert_size_stride(primals_966, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_967, (2304, ), (1, ))
    assert_size_stride(primals_968, (2304, ), (1, ))
    assert_size_stride(primals_969, (2304, ), (1, ))
    assert_size_stride(primals_970, (2304, ), (1, ))
    assert_size_stride(primals_971, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_972, (2304, ), (1, ))
    assert_size_stride(primals_973, (2304, ), (1, ))
    assert_size_stride(primals_974, (2304, ), (1, ))
    assert_size_stride(primals_975, (2304, ), (1, ))
    assert_size_stride(primals_976, (96, 2304), (2304, 1))
    assert_size_stride(primals_977, (96, ), (1, ))
    assert_size_stride(primals_978, (2304, 96), (96, 1))
    assert_size_stride(primals_979, (2304, ), (1, ))
    assert_size_stride(primals_980, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_981, (384, ), (1, ))
    assert_size_stride(primals_982, (384, ), (1, ))
    assert_size_stride(primals_983, (384, ), (1, ))
    assert_size_stride(primals_984, (384, ), (1, ))
    assert_size_stride(primals_985, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_986, (2304, ), (1, ))
    assert_size_stride(primals_987, (2304, ), (1, ))
    assert_size_stride(primals_988, (2304, ), (1, ))
    assert_size_stride(primals_989, (2304, ), (1, ))
    assert_size_stride(primals_990, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_991, (2304, ), (1, ))
    assert_size_stride(primals_992, (2304, ), (1, ))
    assert_size_stride(primals_993, (2304, ), (1, ))
    assert_size_stride(primals_994, (2304, ), (1, ))
    assert_size_stride(primals_995, (96, 2304), (2304, 1))
    assert_size_stride(primals_996, (96, ), (1, ))
    assert_size_stride(primals_997, (2304, 96), (96, 1))
    assert_size_stride(primals_998, (2304, ), (1, ))
    assert_size_stride(primals_999, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1000, (384, ), (1, ))
    assert_size_stride(primals_1001, (384, ), (1, ))
    assert_size_stride(primals_1002, (384, ), (1, ))
    assert_size_stride(primals_1003, (384, ), (1, ))
    assert_size_stride(primals_1004, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1005, (2304, ), (1, ))
    assert_size_stride(primals_1006, (2304, ), (1, ))
    assert_size_stride(primals_1007, (2304, ), (1, ))
    assert_size_stride(primals_1008, (2304, ), (1, ))
    assert_size_stride(primals_1009, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1010, (2304, ), (1, ))
    assert_size_stride(primals_1011, (2304, ), (1, ))
    assert_size_stride(primals_1012, (2304, ), (1, ))
    assert_size_stride(primals_1013, (2304, ), (1, ))
    assert_size_stride(primals_1014, (96, 2304), (2304, 1))
    assert_size_stride(primals_1015, (96, ), (1, ))
    assert_size_stride(primals_1016, (2304, 96), (96, 1))
    assert_size_stride(primals_1017, (2304, ), (1, ))
    assert_size_stride(primals_1018, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1019, (384, ), (1, ))
    assert_size_stride(primals_1020, (384, ), (1, ))
    assert_size_stride(primals_1021, (384, ), (1, ))
    assert_size_stride(primals_1022, (384, ), (1, ))
    assert_size_stride(primals_1023, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1024, (2304, ), (1, ))
    assert_size_stride(primals_1025, (2304, ), (1, ))
    assert_size_stride(primals_1026, (2304, ), (1, ))
    assert_size_stride(primals_1027, (2304, ), (1, ))
    assert_size_stride(primals_1028, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1029, (2304, ), (1, ))
    assert_size_stride(primals_1030, (2304, ), (1, ))
    assert_size_stride(primals_1031, (2304, ), (1, ))
    assert_size_stride(primals_1032, (2304, ), (1, ))
    assert_size_stride(primals_1033, (96, 2304), (2304, 1))
    assert_size_stride(primals_1034, (96, ), (1, ))
    assert_size_stride(primals_1035, (2304, 96), (96, 1))
    assert_size_stride(primals_1036, (2304, ), (1, ))
    assert_size_stride(primals_1037, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1038, (384, ), (1, ))
    assert_size_stride(primals_1039, (384, ), (1, ))
    assert_size_stride(primals_1040, (384, ), (1, ))
    assert_size_stride(primals_1041, (384, ), (1, ))
    assert_size_stride(primals_1042, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1043, (2304, ), (1, ))
    assert_size_stride(primals_1044, (2304, ), (1, ))
    assert_size_stride(primals_1045, (2304, ), (1, ))
    assert_size_stride(primals_1046, (2304, ), (1, ))
    assert_size_stride(primals_1047, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1048, (2304, ), (1, ))
    assert_size_stride(primals_1049, (2304, ), (1, ))
    assert_size_stride(primals_1050, (2304, ), (1, ))
    assert_size_stride(primals_1051, (2304, ), (1, ))
    assert_size_stride(primals_1052, (96, 2304), (2304, 1))
    assert_size_stride(primals_1053, (96, ), (1, ))
    assert_size_stride(primals_1054, (2304, 96), (96, 1))
    assert_size_stride(primals_1055, (2304, ), (1, ))
    assert_size_stride(primals_1056, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1057, (384, ), (1, ))
    assert_size_stride(primals_1058, (384, ), (1, ))
    assert_size_stride(primals_1059, (384, ), (1, ))
    assert_size_stride(primals_1060, (384, ), (1, ))
    assert_size_stride(primals_1061, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1062, (2304, ), (1, ))
    assert_size_stride(primals_1063, (2304, ), (1, ))
    assert_size_stride(primals_1064, (2304, ), (1, ))
    assert_size_stride(primals_1065, (2304, ), (1, ))
    assert_size_stride(primals_1066, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1067, (2304, ), (1, ))
    assert_size_stride(primals_1068, (2304, ), (1, ))
    assert_size_stride(primals_1069, (2304, ), (1, ))
    assert_size_stride(primals_1070, (2304, ), (1, ))
    assert_size_stride(primals_1071, (96, 2304), (2304, 1))
    assert_size_stride(primals_1072, (96, ), (1, ))
    assert_size_stride(primals_1073, (2304, 96), (96, 1))
    assert_size_stride(primals_1074, (2304, ), (1, ))
    assert_size_stride(primals_1075, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1076, (384, ), (1, ))
    assert_size_stride(primals_1077, (384, ), (1, ))
    assert_size_stride(primals_1078, (384, ), (1, ))
    assert_size_stride(primals_1079, (384, ), (1, ))
    assert_size_stride(primals_1080, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1081, (2304, ), (1, ))
    assert_size_stride(primals_1082, (2304, ), (1, ))
    assert_size_stride(primals_1083, (2304, ), (1, ))
    assert_size_stride(primals_1084, (2304, ), (1, ))
    assert_size_stride(primals_1085, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1086, (2304, ), (1, ))
    assert_size_stride(primals_1087, (2304, ), (1, ))
    assert_size_stride(primals_1088, (2304, ), (1, ))
    assert_size_stride(primals_1089, (2304, ), (1, ))
    assert_size_stride(primals_1090, (96, 2304), (2304, 1))
    assert_size_stride(primals_1091, (96, ), (1, ))
    assert_size_stride(primals_1092, (2304, 96), (96, 1))
    assert_size_stride(primals_1093, (2304, ), (1, ))
    assert_size_stride(primals_1094, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1095, (384, ), (1, ))
    assert_size_stride(primals_1096, (384, ), (1, ))
    assert_size_stride(primals_1097, (384, ), (1, ))
    assert_size_stride(primals_1098, (384, ), (1, ))
    assert_size_stride(primals_1099, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1100, (2304, ), (1, ))
    assert_size_stride(primals_1101, (2304, ), (1, ))
    assert_size_stride(primals_1102, (2304, ), (1, ))
    assert_size_stride(primals_1103, (2304, ), (1, ))
    assert_size_stride(primals_1104, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1105, (2304, ), (1, ))
    assert_size_stride(primals_1106, (2304, ), (1, ))
    assert_size_stride(primals_1107, (2304, ), (1, ))
    assert_size_stride(primals_1108, (2304, ), (1, ))
    assert_size_stride(primals_1109, (96, 2304), (2304, 1))
    assert_size_stride(primals_1110, (96, ), (1, ))
    assert_size_stride(primals_1111, (2304, 96), (96, 1))
    assert_size_stride(primals_1112, (2304, ), (1, ))
    assert_size_stride(primals_1113, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1114, (384, ), (1, ))
    assert_size_stride(primals_1115, (384, ), (1, ))
    assert_size_stride(primals_1116, (384, ), (1, ))
    assert_size_stride(primals_1117, (384, ), (1, ))
    assert_size_stride(primals_1118, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1119, (2304, ), (1, ))
    assert_size_stride(primals_1120, (2304, ), (1, ))
    assert_size_stride(primals_1121, (2304, ), (1, ))
    assert_size_stride(primals_1122, (2304, ), (1, ))
    assert_size_stride(primals_1123, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1124, (2304, ), (1, ))
    assert_size_stride(primals_1125, (2304, ), (1, ))
    assert_size_stride(primals_1126, (2304, ), (1, ))
    assert_size_stride(primals_1127, (2304, ), (1, ))
    assert_size_stride(primals_1128, (96, 2304), (2304, 1))
    assert_size_stride(primals_1129, (96, ), (1, ))
    assert_size_stride(primals_1130, (2304, 96), (96, 1))
    assert_size_stride(primals_1131, (2304, ), (1, ))
    assert_size_stride(primals_1132, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1133, (384, ), (1, ))
    assert_size_stride(primals_1134, (384, ), (1, ))
    assert_size_stride(primals_1135, (384, ), (1, ))
    assert_size_stride(primals_1136, (384, ), (1, ))
    assert_size_stride(primals_1137, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1138, (2304, ), (1, ))
    assert_size_stride(primals_1139, (2304, ), (1, ))
    assert_size_stride(primals_1140, (2304, ), (1, ))
    assert_size_stride(primals_1141, (2304, ), (1, ))
    assert_size_stride(primals_1142, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1143, (2304, ), (1, ))
    assert_size_stride(primals_1144, (2304, ), (1, ))
    assert_size_stride(primals_1145, (2304, ), (1, ))
    assert_size_stride(primals_1146, (2304, ), (1, ))
    assert_size_stride(primals_1147, (96, 2304), (2304, 1))
    assert_size_stride(primals_1148, (96, ), (1, ))
    assert_size_stride(primals_1149, (2304, 96), (96, 1))
    assert_size_stride(primals_1150, (2304, ), (1, ))
    assert_size_stride(primals_1151, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1152, (384, ), (1, ))
    assert_size_stride(primals_1153, (384, ), (1, ))
    assert_size_stride(primals_1154, (384, ), (1, ))
    assert_size_stride(primals_1155, (384, ), (1, ))
    assert_size_stride(primals_1156, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1157, (2304, ), (1, ))
    assert_size_stride(primals_1158, (2304, ), (1, ))
    assert_size_stride(primals_1159, (2304, ), (1, ))
    assert_size_stride(primals_1160, (2304, ), (1, ))
    assert_size_stride(primals_1161, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1162, (2304, ), (1, ))
    assert_size_stride(primals_1163, (2304, ), (1, ))
    assert_size_stride(primals_1164, (2304, ), (1, ))
    assert_size_stride(primals_1165, (2304, ), (1, ))
    assert_size_stride(primals_1166, (96, 2304), (2304, 1))
    assert_size_stride(primals_1167, (96, ), (1, ))
    assert_size_stride(primals_1168, (2304, 96), (96, 1))
    assert_size_stride(primals_1169, (2304, ), (1, ))
    assert_size_stride(primals_1170, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1171, (384, ), (1, ))
    assert_size_stride(primals_1172, (384, ), (1, ))
    assert_size_stride(primals_1173, (384, ), (1, ))
    assert_size_stride(primals_1174, (384, ), (1, ))
    assert_size_stride(primals_1175, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1176, (2304, ), (1, ))
    assert_size_stride(primals_1177, (2304, ), (1, ))
    assert_size_stride(primals_1178, (2304, ), (1, ))
    assert_size_stride(primals_1179, (2304, ), (1, ))
    assert_size_stride(primals_1180, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1181, (2304, ), (1, ))
    assert_size_stride(primals_1182, (2304, ), (1, ))
    assert_size_stride(primals_1183, (2304, ), (1, ))
    assert_size_stride(primals_1184, (2304, ), (1, ))
    assert_size_stride(primals_1185, (96, 2304), (2304, 1))
    assert_size_stride(primals_1186, (96, ), (1, ))
    assert_size_stride(primals_1187, (2304, 96), (96, 1))
    assert_size_stride(primals_1188, (2304, ), (1, ))
    assert_size_stride(primals_1189, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1190, (384, ), (1, ))
    assert_size_stride(primals_1191, (384, ), (1, ))
    assert_size_stride(primals_1192, (384, ), (1, ))
    assert_size_stride(primals_1193, (384, ), (1, ))
    assert_size_stride(primals_1194, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1195, (2304, ), (1, ))
    assert_size_stride(primals_1196, (2304, ), (1, ))
    assert_size_stride(primals_1197, (2304, ), (1, ))
    assert_size_stride(primals_1198, (2304, ), (1, ))
    assert_size_stride(primals_1199, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1200, (2304, ), (1, ))
    assert_size_stride(primals_1201, (2304, ), (1, ))
    assert_size_stride(primals_1202, (2304, ), (1, ))
    assert_size_stride(primals_1203, (2304, ), (1, ))
    assert_size_stride(primals_1204, (96, 2304), (2304, 1))
    assert_size_stride(primals_1205, (96, ), (1, ))
    assert_size_stride(primals_1206, (2304, 96), (96, 1))
    assert_size_stride(primals_1207, (2304, ), (1, ))
    assert_size_stride(primals_1208, (384, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1209, (384, ), (1, ))
    assert_size_stride(primals_1210, (384, ), (1, ))
    assert_size_stride(primals_1211, (384, ), (1, ))
    assert_size_stride(primals_1212, (384, ), (1, ))
    assert_size_stride(primals_1213, (2304, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_1214, (2304, ), (1, ))
    assert_size_stride(primals_1215, (2304, ), (1, ))
    assert_size_stride(primals_1216, (2304, ), (1, ))
    assert_size_stride(primals_1217, (2304, ), (1, ))
    assert_size_stride(primals_1218, (2304, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1219, (2304, ), (1, ))
    assert_size_stride(primals_1220, (2304, ), (1, ))
    assert_size_stride(primals_1221, (2304, ), (1, ))
    assert_size_stride(primals_1222, (2304, ), (1, ))
    assert_size_stride(primals_1223, (96, 2304), (2304, 1))
    assert_size_stride(primals_1224, (96, ), (1, ))
    assert_size_stride(primals_1225, (2304, 96), (96, 1))
    assert_size_stride(primals_1226, (2304, ), (1, ))
    assert_size_stride(primals_1227, (640, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_1228, (640, ), (1, ))
    assert_size_stride(primals_1229, (640, ), (1, ))
    assert_size_stride(primals_1230, (640, ), (1, ))
    assert_size_stride(primals_1231, (640, ), (1, ))
    assert_size_stride(primals_1232, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1233, (3840, ), (1, ))
    assert_size_stride(primals_1234, (3840, ), (1, ))
    assert_size_stride(primals_1235, (3840, ), (1, ))
    assert_size_stride(primals_1236, (3840, ), (1, ))
    assert_size_stride(primals_1237, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1238, (3840, ), (1, ))
    assert_size_stride(primals_1239, (3840, ), (1, ))
    assert_size_stride(primals_1240, (3840, ), (1, ))
    assert_size_stride(primals_1241, (3840, ), (1, ))
    assert_size_stride(primals_1242, (160, 3840), (3840, 1))
    assert_size_stride(primals_1243, (160, ), (1, ))
    assert_size_stride(primals_1244, (3840, 160), (160, 1))
    assert_size_stride(primals_1245, (3840, ), (1, ))
    assert_size_stride(primals_1246, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1247, (640, ), (1, ))
    assert_size_stride(primals_1248, (640, ), (1, ))
    assert_size_stride(primals_1249, (640, ), (1, ))
    assert_size_stride(primals_1250, (640, ), (1, ))
    assert_size_stride(primals_1251, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1252, (3840, ), (1, ))
    assert_size_stride(primals_1253, (3840, ), (1, ))
    assert_size_stride(primals_1254, (3840, ), (1, ))
    assert_size_stride(primals_1255, (3840, ), (1, ))
    assert_size_stride(primals_1256, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1257, (3840, ), (1, ))
    assert_size_stride(primals_1258, (3840, ), (1, ))
    assert_size_stride(primals_1259, (3840, ), (1, ))
    assert_size_stride(primals_1260, (3840, ), (1, ))
    assert_size_stride(primals_1261, (160, 3840), (3840, 1))
    assert_size_stride(primals_1262, (160, ), (1, ))
    assert_size_stride(primals_1263, (3840, 160), (160, 1))
    assert_size_stride(primals_1264, (3840, ), (1, ))
    assert_size_stride(primals_1265, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1266, (640, ), (1, ))
    assert_size_stride(primals_1267, (640, ), (1, ))
    assert_size_stride(primals_1268, (640, ), (1, ))
    assert_size_stride(primals_1269, (640, ), (1, ))
    assert_size_stride(primals_1270, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1271, (3840, ), (1, ))
    assert_size_stride(primals_1272, (3840, ), (1, ))
    assert_size_stride(primals_1273, (3840, ), (1, ))
    assert_size_stride(primals_1274, (3840, ), (1, ))
    assert_size_stride(primals_1275, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1276, (3840, ), (1, ))
    assert_size_stride(primals_1277, (3840, ), (1, ))
    assert_size_stride(primals_1278, (3840, ), (1, ))
    assert_size_stride(primals_1279, (3840, ), (1, ))
    assert_size_stride(primals_1280, (160, 3840), (3840, 1))
    assert_size_stride(primals_1281, (160, ), (1, ))
    assert_size_stride(primals_1282, (3840, 160), (160, 1))
    assert_size_stride(primals_1283, (3840, ), (1, ))
    assert_size_stride(primals_1284, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1285, (640, ), (1, ))
    assert_size_stride(primals_1286, (640, ), (1, ))
    assert_size_stride(primals_1287, (640, ), (1, ))
    assert_size_stride(primals_1288, (640, ), (1, ))
    assert_size_stride(primals_1289, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1290, (3840, ), (1, ))
    assert_size_stride(primals_1291, (3840, ), (1, ))
    assert_size_stride(primals_1292, (3840, ), (1, ))
    assert_size_stride(primals_1293, (3840, ), (1, ))
    assert_size_stride(primals_1294, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1295, (3840, ), (1, ))
    assert_size_stride(primals_1296, (3840, ), (1, ))
    assert_size_stride(primals_1297, (3840, ), (1, ))
    assert_size_stride(primals_1298, (3840, ), (1, ))
    assert_size_stride(primals_1299, (160, 3840), (3840, 1))
    assert_size_stride(primals_1300, (160, ), (1, ))
    assert_size_stride(primals_1301, (3840, 160), (160, 1))
    assert_size_stride(primals_1302, (3840, ), (1, ))
    assert_size_stride(primals_1303, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1304, (640, ), (1, ))
    assert_size_stride(primals_1305, (640, ), (1, ))
    assert_size_stride(primals_1306, (640, ), (1, ))
    assert_size_stride(primals_1307, (640, ), (1, ))
    assert_size_stride(primals_1308, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1309, (3840, ), (1, ))
    assert_size_stride(primals_1310, (3840, ), (1, ))
    assert_size_stride(primals_1311, (3840, ), (1, ))
    assert_size_stride(primals_1312, (3840, ), (1, ))
    assert_size_stride(primals_1313, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1314, (3840, ), (1, ))
    assert_size_stride(primals_1315, (3840, ), (1, ))
    assert_size_stride(primals_1316, (3840, ), (1, ))
    assert_size_stride(primals_1317, (3840, ), (1, ))
    assert_size_stride(primals_1318, (160, 3840), (3840, 1))
    assert_size_stride(primals_1319, (160, ), (1, ))
    assert_size_stride(primals_1320, (3840, 160), (160, 1))
    assert_size_stride(primals_1321, (3840, ), (1, ))
    assert_size_stride(primals_1322, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1323, (640, ), (1, ))
    assert_size_stride(primals_1324, (640, ), (1, ))
    assert_size_stride(primals_1325, (640, ), (1, ))
    assert_size_stride(primals_1326, (640, ), (1, ))
    assert_size_stride(primals_1327, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1328, (3840, ), (1, ))
    assert_size_stride(primals_1329, (3840, ), (1, ))
    assert_size_stride(primals_1330, (3840, ), (1, ))
    assert_size_stride(primals_1331, (3840, ), (1, ))
    assert_size_stride(primals_1332, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1333, (3840, ), (1, ))
    assert_size_stride(primals_1334, (3840, ), (1, ))
    assert_size_stride(primals_1335, (3840, ), (1, ))
    assert_size_stride(primals_1336, (3840, ), (1, ))
    assert_size_stride(primals_1337, (160, 3840), (3840, 1))
    assert_size_stride(primals_1338, (160, ), (1, ))
    assert_size_stride(primals_1339, (3840, 160), (160, 1))
    assert_size_stride(primals_1340, (3840, ), (1, ))
    assert_size_stride(primals_1341, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1342, (640, ), (1, ))
    assert_size_stride(primals_1343, (640, ), (1, ))
    assert_size_stride(primals_1344, (640, ), (1, ))
    assert_size_stride(primals_1345, (640, ), (1, ))
    assert_size_stride(primals_1346, (1792, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1347, (1792, ), (1, ))
    assert_size_stride(primals_1348, (1792, ), (1, ))
    assert_size_stride(primals_1349, (1792, ), (1, ))
    assert_size_stride(primals_1350, (1792, ), (1, ))
    assert_size_stride(primals_1351, (1000, 1792), (1792, 1))
    assert_size_stride(primals_1352, (1000, ), (1, ))
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
        buf3 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_17, buf3, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_17
        buf4 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_27, buf4, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_27
        buf5 = empty_strided_cuda((32, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_37, buf5, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_37
        buf6 = empty_strided_cuda((128, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_47, buf6, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_47
        buf7 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_57, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_57
        buf8 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_67, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
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
        buf12 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_107, buf12, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_107
        buf13 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_117, buf13, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_117
        buf14 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_127, buf14, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_127
        buf15 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_137, buf15, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_137
        buf16 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_147, buf16, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_147
        buf17 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_157, buf17, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_157
        buf18 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_167, buf18, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_167
        buf19 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_177, buf19, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_177
        # Topologically Sorted Source Nodes: [features_0_0], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf21 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf22 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [features_0_1, sigmoid_1, mul_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7.run(buf22, buf20, primals_3, primals_4, primals_5, primals_6, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_0], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf24 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf25 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [features_1_conv_1, sigmoid_2, mul_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7.run(buf25, buf23, primals_8, primals_9, primals_10, primals_11, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_3], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf27 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_1_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf26, primals_13, primals_14, primals_15, primals_16, buf27, 131072, grid=grid(131072), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [features_2_conv_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf29 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf30 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [features_2_conv_1, sigmoid_3, mul_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9.run(buf30, buf28, primals_18, primals_19, primals_20, primals_21, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_2_conv_3], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf32 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_2_conv_4, add_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf27, buf31, primals_23, primals_24, primals_25, primals_26, buf32, 131072, grid=grid(131072), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [features_3_conv_0], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf34 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [features_3_conv_1, sigmoid_4, mul_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9.run(buf35, buf33, primals_28, primals_29, primals_30, primals_31, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_3_conv_3], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf37 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_3_conv_4, add_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf32, buf36, primals_33, primals_34, primals_35, primals_36, buf37, 131072, grid=grid(131072), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [features_4_conv_0], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf39 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf40 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [features_4_conv_1, sigmoid_5, mul_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9.run(buf40, buf38, primals_38, primals_39, primals_40, primals_41, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_4_conv_3], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf42 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_4_conv_4, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf37, buf41, primals_43, primals_44, primals_45, primals_46, buf42, 131072, grid=grid(131072), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [features_5_conv_0], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf44 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf45 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [features_5_conv_1, sigmoid_6, mul_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_11.run(buf45, buf43, primals_48, primals_49, primals_50, primals_51, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_5_conv_3], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf47 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_5_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf46, primals_53, primals_54, primals_55, primals_56, buf47, 65536, grid=grid(65536), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [features_6_conv_0], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf49 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf50 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [features_6_conv_1, sigmoid_7, mul_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf50, buf48, primals_58, primals_59, primals_60, primals_61, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_6_conv_3], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf52 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_6_conv_4, add_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf47, buf51, primals_63, primals_64, primals_65, primals_66, buf52, 65536, grid=grid(65536), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [features_7_conv_0], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf54 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf55 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [features_7_conv_1, sigmoid_8, mul_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf55, buf53, primals_68, primals_69, primals_70, primals_71, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_7_conv_3], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf57 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_7_conv_4, add_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf52, buf56, primals_73, primals_74, primals_75, primals_76, buf57, 65536, grid=grid(65536), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [features_8_conv_0], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf59 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf60 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [features_8_conv_1, sigmoid_9, mul_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf60, buf58, primals_78, primals_79, primals_80, primals_81, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_8_conv_3], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf62 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_8_conv_4, add_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf57, buf61, primals_83, primals_84, primals_85, primals_86, buf62, 65536, grid=grid(65536), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [features_9_conv_0], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf64 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf65 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [features_9_conv_1, sigmoid_10, mul_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf65, buf63, primals_88, primals_89, primals_90, primals_91, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_9_conv_3], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf67 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_9_conv_4, add_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf62, buf66, primals_93, primals_94, primals_95, primals_96, buf67, 65536, grid=grid(65536), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [features_10_conv_0], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf69 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf70 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [features_10_conv_1, sigmoid_11, mul_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf70, buf68, primals_98, primals_99, primals_100, primals_101, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_10_conv_3], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf72 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_10_conv_4, add_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf67, buf71, primals_103, primals_104, primals_105, primals_106, buf72, 65536, grid=grid(65536), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [features_11_conv_0], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf74 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf75 = buf74; del buf74  # reuse
        # Topologically Sorted Source Nodes: [features_11_conv_1, sigmoid_12, mul_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf75, buf73, primals_108, primals_109, primals_110, primals_111, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_11_conv_3], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf77 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_11_conv_4, add_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf72, buf76, primals_113, primals_114, primals_115, primals_116, buf77, 65536, grid=grid(65536), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [features_12_conv_0], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, buf13, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf79 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf80 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [features_12_conv_1, sigmoid_13, mul_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_15.run(buf80, buf78, primals_118, primals_119, primals_120, primals_121, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [features_12_conv_3], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf82 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_12_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf81, primals_123, primals_124, primals_125, primals_126, buf82, 24576, grid=grid(24576), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [features_13_conv_0], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf84 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf85 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [features_13_conv_1, sigmoid_14, mul_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf85, buf83, primals_128, primals_129, primals_130, primals_131, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_13_conv_3], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf87 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_13_conv_4, add_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf82, buf86, primals_133, primals_134, primals_135, primals_136, buf87, 24576, grid=grid(24576), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [features_14_conv_0], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf89 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf90 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [features_14_conv_1, sigmoid_15, mul_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf90, buf88, primals_138, primals_139, primals_140, primals_141, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_14_conv_3], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf92 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_4, add_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf87, buf91, primals_143, primals_144, primals_145, primals_146, buf92, 24576, grid=grid(24576), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [features_15_conv_0], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf94 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf95 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [features_15_conv_1, sigmoid_16, mul_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf95, buf93, primals_148, primals_149, primals_150, primals_151, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_15_conv_3], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf97 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_4, add_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf92, buf96, primals_153, primals_154, primals_155, primals_156, buf97, 24576, grid=grid(24576), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [features_16_conv_0], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf99 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf100 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [features_16_conv_1, sigmoid_17, mul_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf100, buf98, primals_158, primals_159, primals_160, primals_161, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_16_conv_3], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf102 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_4, add_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf97, buf101, primals_163, primals_164, primals_165, primals_166, buf102, 24576, grid=grid(24576), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [features_17_conv_0], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf104 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [features_17_conv_1, sigmoid_18, mul_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf105, buf103, primals_168, primals_169, primals_170, primals_171, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_17_conv_3], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf107 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_4, add_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf102, buf106, primals_173, primals_174, primals_175, primals_176, buf107, 24576, grid=grid(24576), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [features_18_conv_0], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf109 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf110 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [features_18_conv_1, sigmoid_19, mul_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf110, buf108, primals_178, primals_179, primals_180, primals_181, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_18_conv_3], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf112 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_4, add_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf107, buf111, primals_183, primals_184, primals_185, primals_186, buf112, 24576, grid=grid(24576), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [features_19_conv_0], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_187, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf114 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf115 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [features_19_conv_1, sigmoid_20, mul_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf115, buf113, primals_188, primals_189, primals_190, primals_191, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_19_conv_3], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_192, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf116, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf117 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf116, primals_193, primals_194, primals_195, primals_196, buf117, 24576, grid=grid(24576), stream=stream0)
        buf118 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 1536, 1536), torch.float32)
        buf119 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_21, mul_21, features_19_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_20.run(buf119, buf117, 1536, 16, grid=grid(1536), stream=stream0)
        buf120 = empty_strided_cuda((4, 24), (24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_198, reinterpret_tensor(buf119, (4, 384), (384, 1), 0), reinterpret_tensor(primals_197, (384, 24), (1, 384), 0), alpha=1, beta=1, out=buf120)
        del primals_198
        buf121 = empty_strided_cuda((4, 24), (24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_22, mul_22], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_21.run(buf120, buf121, 96, grid=grid(96), stream=stream0)
        buf122 = empty_strided_cuda((4, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_200, buf121, reinterpret_tensor(primals_199, (24, 384), (1, 24), 0), alpha=1, beta=1, out=buf122)
        del primals_200
        buf123 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_21, mul_21, mul_23], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_22.run(buf123, buf122, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_19_conv_7], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_201, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf125 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf124, primals_202, primals_203, primals_204, primals_205, buf125, 12288, grid=grid(12288), stream=stream0)
        del primals_205
        # Topologically Sorted Source Nodes: [features_20_conv_0], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf127 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf128 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [features_20_conv_1, sigmoid_25, mul_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf128, buf126, primals_207, primals_208, primals_209, primals_210, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_20_conv_3], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_211, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf129, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf130 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf129, primals_212, primals_213, primals_214, primals_215, buf130, 49152, grid=grid(49152), stream=stream0)
        buf131 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_27, mul_25, features_20_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf132, buf130, 3072, 16, grid=grid(3072), stream=stream0)
        buf133 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_217, reinterpret_tensor(buf132, (4, 768), (768, 1), 0), reinterpret_tensor(primals_216, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf133)
        del primals_217
        buf134 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_29, mul_26], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf133, buf134, 192, grid=grid(192), stream=stream0)
        buf135 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_219, buf134, reinterpret_tensor(primals_218, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf135)
        del primals_219
        buf136 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_27, mul_25, mul_27], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf136, buf135, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_20_conv_7], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf138 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_8, add_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf125, buf137, primals_221, primals_222, primals_223, primals_224, buf138, 12288, grid=grid(12288), stream=stream0)
        del primals_224
        # Topologically Sorted Source Nodes: [features_21_conv_0], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf140 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf141 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [features_21_conv_1, sigmoid_32, mul_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf141, buf139, primals_226, primals_227, primals_228, primals_229, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_3], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf141, primals_230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf142, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf143 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf142, primals_231, primals_232, primals_233, primals_234, buf143, 49152, grid=grid(49152), stream=stream0)
        buf144 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf145 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_34, mul_29, features_21_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf145, buf143, 3072, 16, grid=grid(3072), stream=stream0)
        buf146 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_236, reinterpret_tensor(buf145, (4, 768), (768, 1), 0), reinterpret_tensor(primals_235, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf146)
        del primals_236
        buf147 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_36, mul_30], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf146, buf147, 192, grid=grid(192), stream=stream0)
        buf148 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_238, buf147, reinterpret_tensor(primals_237, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf148)
        del primals_238
        buf149 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_34, mul_29, mul_31], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf149, buf148, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_7], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_239, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf151 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_8, add_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf138, buf150, primals_240, primals_241, primals_242, primals_243, buf151, 12288, grid=grid(12288), stream=stream0)
        del primals_243
        # Topologically Sorted Source Nodes: [features_22_conv_0], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf153 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [features_22_conv_1, sigmoid_39, mul_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf154, buf152, primals_245, primals_246, primals_247, primals_248, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_3], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_249, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf155, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf156 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf155, primals_250, primals_251, primals_252, primals_253, buf156, 49152, grid=grid(49152), stream=stream0)
        buf157 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf158 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_41, mul_33, features_22_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf158, buf156, 3072, 16, grid=grid(3072), stream=stream0)
        buf159 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_255, reinterpret_tensor(buf158, (4, 768), (768, 1), 0), reinterpret_tensor(primals_254, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf159)
        del primals_255
        buf160 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_43, mul_34], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf159, buf160, 192, grid=grid(192), stream=stream0)
        buf161 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_257, buf160, reinterpret_tensor(primals_256, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf161)
        del primals_257
        buf162 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_41, mul_33, mul_35], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf162, buf161, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_7], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf164 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_8, add_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf151, buf163, primals_259, primals_260, primals_261, primals_262, buf164, 12288, grid=grid(12288), stream=stream0)
        del primals_262
        # Topologically Sorted Source Nodes: [features_23_conv_0], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf166 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf167 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [features_23_conv_1, sigmoid_46, mul_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf167, buf165, primals_264, primals_265, primals_266, primals_267, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_3], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf167, primals_268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf168, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf169 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf168, primals_269, primals_270, primals_271, primals_272, buf169, 49152, grid=grid(49152), stream=stream0)
        buf170 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf171 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_48, mul_37, features_23_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf171, buf169, 3072, 16, grid=grid(3072), stream=stream0)
        buf172 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_274, reinterpret_tensor(buf171, (4, 768), (768, 1), 0), reinterpret_tensor(primals_273, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf172)
        del primals_274
        buf173 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_50, mul_38], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf172, buf173, 192, grid=grid(192), stream=stream0)
        buf174 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_276, buf173, reinterpret_tensor(primals_275, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf174)
        del primals_276
        buf175 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_48, mul_37, mul_39], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf175, buf174, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_7], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf177 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_8, add_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf164, buf176, primals_278, primals_279, primals_280, primals_281, buf177, 12288, grid=grid(12288), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [features_24_conv_0], Original ATen: [aten.convolution]
        buf178 = extern_kernels.convolution(buf177, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf179 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf180 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [features_24_conv_1, sigmoid_53, mul_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf180, buf178, primals_283, primals_284, primals_285, primals_286, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_3], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf181, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf182 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf181, primals_288, primals_289, primals_290, primals_291, buf182, 49152, grid=grid(49152), stream=stream0)
        buf183 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf184 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_55, mul_41, features_24_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf184, buf182, 3072, 16, grid=grid(3072), stream=stream0)
        buf185 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_293, reinterpret_tensor(buf184, (4, 768), (768, 1), 0), reinterpret_tensor(primals_292, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf185)
        del primals_293
        buf186 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_57, mul_42], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf185, buf186, 192, grid=grid(192), stream=stream0)
        buf187 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_295, buf186, reinterpret_tensor(primals_294, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf187)
        del primals_295
        buf188 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_55, mul_41, mul_43], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf188, buf187, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_7], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_296, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf190 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_8, add_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf177, buf189, primals_297, primals_298, primals_299, primals_300, buf190, 12288, grid=grid(12288), stream=stream0)
        del primals_300
        # Topologically Sorted Source Nodes: [features_25_conv_0], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf192 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf193 = buf192; del buf192  # reuse
        # Topologically Sorted Source Nodes: [features_25_conv_1, sigmoid_60, mul_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf193, buf191, primals_302, primals_303, primals_304, primals_305, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_3], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf194, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf195 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf194, primals_307, primals_308, primals_309, primals_310, buf195, 49152, grid=grid(49152), stream=stream0)
        buf196 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf197 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_62, mul_45, features_25_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf197, buf195, 3072, 16, grid=grid(3072), stream=stream0)
        buf198 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_312, reinterpret_tensor(buf197, (4, 768), (768, 1), 0), reinterpret_tensor(primals_311, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf198)
        del primals_312
        buf199 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_64, mul_46], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf198, buf199, 192, grid=grid(192), stream=stream0)
        buf200 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_314, buf199, reinterpret_tensor(primals_313, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf200)
        del primals_314
        buf201 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_62, mul_45, mul_47], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf201, buf200, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_7], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf203 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_8, add_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf190, buf202, primals_316, primals_317, primals_318, primals_319, buf203, 12288, grid=grid(12288), stream=stream0)
        del primals_319
        # Topologically Sorted Source Nodes: [features_26_conv_0], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_320, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf205 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf206 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [features_26_conv_1, sigmoid_67, mul_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf206, buf204, primals_321, primals_322, primals_323, primals_324, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_3], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_325, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf207, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf208 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf207, primals_326, primals_327, primals_328, primals_329, buf208, 49152, grid=grid(49152), stream=stream0)
        buf209 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_69, mul_49, features_26_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf210, buf208, 3072, 16, grid=grid(3072), stream=stream0)
        buf211 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_331, reinterpret_tensor(buf210, (4, 768), (768, 1), 0), reinterpret_tensor(primals_330, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf211)
        del primals_331
        buf212 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_71, mul_50], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf211, buf212, 192, grid=grid(192), stream=stream0)
        buf213 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_333, buf212, reinterpret_tensor(primals_332, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf213)
        del primals_333
        buf214 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_69, mul_49, mul_51], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf214, buf213, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_7], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf216 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_8, add_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf203, buf215, primals_335, primals_336, primals_337, primals_338, buf216, 12288, grid=grid(12288), stream=stream0)
        del primals_338
        # Topologically Sorted Source Nodes: [features_27_conv_0], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_339, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf218 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf219 = buf218; del buf218  # reuse
        # Topologically Sorted Source Nodes: [features_27_conv_1, sigmoid_74, mul_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf219, buf217, primals_340, primals_341, primals_342, primals_343, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_3], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_344, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf220, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf221 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf220, primals_345, primals_346, primals_347, primals_348, buf221, 49152, grid=grid(49152), stream=stream0)
        buf222 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf223 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_76, mul_53, features_27_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf223, buf221, 3072, 16, grid=grid(3072), stream=stream0)
        buf224 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_350, reinterpret_tensor(buf223, (4, 768), (768, 1), 0), reinterpret_tensor(primals_349, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf224)
        del primals_350
        buf225 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_78, mul_54], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf224, buf225, 192, grid=grid(192), stream=stream0)
        buf226 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_352, buf225, reinterpret_tensor(primals_351, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf226)
        del primals_352
        buf227 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_76, mul_53, mul_55], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf227, buf226, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_7], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_353, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf229 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_8, add_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf216, buf228, primals_354, primals_355, primals_356, primals_357, buf229, 12288, grid=grid(12288), stream=stream0)
        del primals_357
        # Topologically Sorted Source Nodes: [features_28_conv_0], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_358, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf231 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf232 = buf231; del buf231  # reuse
        # Topologically Sorted Source Nodes: [features_28_conv_1, sigmoid_81, mul_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf232, buf230, primals_359, primals_360, primals_361, primals_362, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_3], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_363, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf233, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf234 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf233, primals_364, primals_365, primals_366, primals_367, buf234, 49152, grid=grid(49152), stream=stream0)
        buf235 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf236 = buf235; del buf235  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_83, mul_57, features_28_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf236, buf234, 3072, 16, grid=grid(3072), stream=stream0)
        buf237 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_369, reinterpret_tensor(buf236, (4, 768), (768, 1), 0), reinterpret_tensor(primals_368, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf237)
        del primals_369
        buf238 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_85, mul_58], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf237, buf238, 192, grid=grid(192), stream=stream0)
        buf239 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_371, buf238, reinterpret_tensor(primals_370, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf239)
        del primals_371
        buf240 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_83, mul_57, mul_59], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf240, buf239, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_7], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf242 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_8, add_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf229, buf241, primals_373, primals_374, primals_375, primals_376, buf242, 12288, grid=grid(12288), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [features_29_conv_0], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_377, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 1152, 4, 4), (18432, 1, 4608, 1152))
        buf244 = empty_strided_cuda((4, 1152, 4, 4), (18432, 1, 4608, 1152), torch.float32)
        buf245 = buf244; del buf244  # reuse
        # Topologically Sorted Source Nodes: [features_29_conv_1, sigmoid_88, mul_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_30.run(buf245, buf243, primals_378, primals_379, primals_380, primals_381, 73728, grid=grid(73728), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_3], Original ATen: [aten.convolution]
        buf246 = extern_kernels.convolution(buf245, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf246, (4, 1152, 4, 4), (18432, 1, 4608, 1152))
        buf247 = empty_strided_cuda((4, 1152, 4, 4), (18432, 1, 4608, 1152), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf246, primals_383, primals_384, primals_385, primals_386, buf247, 73728, grid=grid(73728), stream=stream0)
        buf248 = empty_strided_cuda((4, 1152, 1, 1), (1152, 1, 4608, 4608), torch.float32)
        buf249 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_90, mul_61, features_29_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_32.run(buf249, buf247, 4608, 16, grid=grid(4608), stream=stream0)
        buf250 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_388, reinterpret_tensor(buf249, (4, 1152), (1152, 1), 0), reinterpret_tensor(primals_387, (1152, 48), (1, 1152), 0), alpha=1, beta=1, out=buf250)
        del primals_388
        buf251 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_92, mul_62], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf250, buf251, 192, grid=grid(192), stream=stream0)
        buf252 = empty_strided_cuda((4, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_390, buf251, reinterpret_tensor(primals_389, (48, 1152), (1, 48), 0), alpha=1, beta=1, out=buf252)
        del primals_390
        buf253 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_90, mul_61, mul_63], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_33.run(buf253, buf252, 73728, grid=grid(73728), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_7], Original ATen: [aten.convolution]
        buf254 = extern_kernels.convolution(buf253, primals_391, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf255 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf254, primals_392, primals_393, primals_394, primals_395, buf255, 14336, grid=grid(14336), stream=stream0)
        del primals_395
        # Topologically Sorted Source Nodes: [features_30_conv_0], Original ATen: [aten.convolution]
        buf256 = extern_kernels.convolution(buf255, primals_396, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf257 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [features_30_conv_1, sigmoid_95, mul_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf258, buf256, primals_397, primals_398, primals_399, primals_400, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_3], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_401, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf259, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf260 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf259, primals_402, primals_403, primals_404, primals_405, buf260, 86016, grid=grid(86016), stream=stream0)
        buf261 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf262 = buf261; del buf261  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_97, mul_65, features_30_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf262, buf260, 5376, 16, grid=grid(5376), stream=stream0)
        buf263 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_407, reinterpret_tensor(buf262, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_406, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf263)
        del primals_407
        buf264 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_99, mul_66], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf263, buf264, 224, grid=grid(224), stream=stream0)
        buf265 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_409, buf264, reinterpret_tensor(primals_408, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf265)
        del primals_409
        buf266 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_97, mul_65, mul_67], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf266, buf265, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_7], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_410, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf268 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_8, add_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf255, buf267, primals_411, primals_412, primals_413, primals_414, buf268, 14336, grid=grid(14336), stream=stream0)
        del primals_414
        # Topologically Sorted Source Nodes: [features_31_conv_0], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_415, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf269, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf270 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf271 = buf270; del buf270  # reuse
        # Topologically Sorted Source Nodes: [features_31_conv_1, sigmoid_102, mul_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf271, buf269, primals_416, primals_417, primals_418, primals_419, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_3], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_420, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf272, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf273 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf272, primals_421, primals_422, primals_423, primals_424, buf273, 86016, grid=grid(86016), stream=stream0)
        buf274 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf275 = buf274; del buf274  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_104, mul_69, features_31_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf275, buf273, 5376, 16, grid=grid(5376), stream=stream0)
        buf276 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_426, reinterpret_tensor(buf275, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_425, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf276)
        del primals_426
        buf277 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_106, mul_70], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf276, buf277, 224, grid=grid(224), stream=stream0)
        buf278 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_428, buf277, reinterpret_tensor(primals_427, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf278)
        del primals_428
        buf279 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_104, mul_69, mul_71], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf279, buf278, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_7], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_429, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf280, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf281 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_8, add_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf268, buf280, primals_430, primals_431, primals_432, primals_433, buf281, 14336, grid=grid(14336), stream=stream0)
        del primals_433
        # Topologically Sorted Source Nodes: [features_32_conv_0], Original ATen: [aten.convolution]
        buf282 = extern_kernels.convolution(buf281, primals_434, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf283 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf284 = buf283; del buf283  # reuse
        # Topologically Sorted Source Nodes: [features_32_conv_1, sigmoid_109, mul_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf284, buf282, primals_435, primals_436, primals_437, primals_438, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_3], Original ATen: [aten.convolution]
        buf285 = extern_kernels.convolution(buf284, primals_439, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf285, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf286 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf285, primals_440, primals_441, primals_442, primals_443, buf286, 86016, grid=grid(86016), stream=stream0)
        buf287 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf288 = buf287; del buf287  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_111, mul_73, features_32_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf288, buf286, 5376, 16, grid=grid(5376), stream=stream0)
        buf289 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_445, reinterpret_tensor(buf288, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_444, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf289)
        del primals_445
        buf290 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_113, mul_74], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf289, buf290, 224, grid=grid(224), stream=stream0)
        buf291 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_447, buf290, reinterpret_tensor(primals_446, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf291)
        del primals_447
        buf292 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_111, mul_73, mul_75], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf292, buf291, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_7], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_448, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf294 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_8, add_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf281, buf293, primals_449, primals_450, primals_451, primals_452, buf294, 14336, grid=grid(14336), stream=stream0)
        del primals_452
        # Topologically Sorted Source Nodes: [features_33_conv_0], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_453, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf296 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf297 = buf296; del buf296  # reuse
        # Topologically Sorted Source Nodes: [features_33_conv_1, sigmoid_116, mul_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf297, buf295, primals_454, primals_455, primals_456, primals_457, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_3], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_458, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf298, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf299 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf298, primals_459, primals_460, primals_461, primals_462, buf299, 86016, grid=grid(86016), stream=stream0)
        buf300 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf301 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_118, mul_77, features_33_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf301, buf299, 5376, 16, grid=grid(5376), stream=stream0)
        buf302 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_464, reinterpret_tensor(buf301, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_463, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf302)
        del primals_464
        buf303 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_120, mul_78], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf302, buf303, 224, grid=grid(224), stream=stream0)
        buf304 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_466, buf303, reinterpret_tensor(primals_465, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf304)
        del primals_466
        buf305 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_118, mul_77, mul_79], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf305, buf304, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_7], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_467, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf307 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_8, add_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf294, buf306, primals_468, primals_469, primals_470, primals_471, buf307, 14336, grid=grid(14336), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [features_34_conv_0], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf308, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf309 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf310 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [features_34_conv_1, sigmoid_123, mul_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf310, buf308, primals_473, primals_474, primals_475, primals_476, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_3], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_477, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf311, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf312 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf311, primals_478, primals_479, primals_480, primals_481, buf312, 86016, grid=grid(86016), stream=stream0)
        buf313 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf314 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_125, mul_81, features_34_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf314, buf312, 5376, 16, grid=grid(5376), stream=stream0)
        buf315 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_483, reinterpret_tensor(buf314, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_482, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf315)
        del primals_483
        buf316 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_127, mul_82], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf315, buf316, 224, grid=grid(224), stream=stream0)
        buf317 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_485, buf316, reinterpret_tensor(primals_484, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf317)
        del primals_485
        buf318 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_125, mul_81, mul_83], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf318, buf317, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_7], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_486, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf320 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_8, add_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf307, buf319, primals_487, primals_488, primals_489, primals_490, buf320, 14336, grid=grid(14336), stream=stream0)
        del primals_490
        # Topologically Sorted Source Nodes: [features_35_conv_0], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, primals_491, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf322 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf323 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [features_35_conv_1, sigmoid_130, mul_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf323, buf321, primals_492, primals_493, primals_494, primals_495, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_3], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_496, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf324, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf325 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf324, primals_497, primals_498, primals_499, primals_500, buf325, 86016, grid=grid(86016), stream=stream0)
        buf326 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf327 = buf326; del buf326  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_132, mul_85, features_35_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf327, buf325, 5376, 16, grid=grid(5376), stream=stream0)
        buf328 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_502, reinterpret_tensor(buf327, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_501, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf328)
        del primals_502
        buf329 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_134, mul_86], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf328, buf329, 224, grid=grid(224), stream=stream0)
        buf330 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_504, buf329, reinterpret_tensor(primals_503, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf330)
        del primals_504
        buf331 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_132, mul_85, mul_87], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf331, buf330, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_7], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_505, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf333 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_8, add_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf320, buf332, primals_506, primals_507, primals_508, primals_509, buf333, 14336, grid=grid(14336), stream=stream0)
        del primals_509
        # Topologically Sorted Source Nodes: [features_36_conv_0], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_510, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf335 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf336 = buf335; del buf335  # reuse
        # Topologically Sorted Source Nodes: [features_36_conv_1, sigmoid_137, mul_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf336, buf334, primals_511, primals_512, primals_513, primals_514, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_3], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, primals_515, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf337, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf338 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf337, primals_516, primals_517, primals_518, primals_519, buf338, 86016, grid=grid(86016), stream=stream0)
        buf339 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf340 = buf339; del buf339  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_139, mul_89, features_36_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf340, buf338, 5376, 16, grid=grid(5376), stream=stream0)
        buf341 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_521, reinterpret_tensor(buf340, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_520, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf341)
        del primals_521
        buf342 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_141, mul_90], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf341, buf342, 224, grid=grid(224), stream=stream0)
        buf343 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_523, buf342, reinterpret_tensor(primals_522, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf343)
        del primals_523
        buf344 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_139, mul_89, mul_91], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf344, buf343, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_7], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf344, primals_524, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf346 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_8, add_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf333, buf345, primals_525, primals_526, primals_527, primals_528, buf346, 14336, grid=grid(14336), stream=stream0)
        del primals_528
        # Topologically Sorted Source Nodes: [features_37_conv_0], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_529, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf348 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf349 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [features_37_conv_1, sigmoid_144, mul_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf349, buf347, primals_530, primals_531, primals_532, primals_533, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_3], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf349, primals_534, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf350, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf351 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf350, primals_535, primals_536, primals_537, primals_538, buf351, 86016, grid=grid(86016), stream=stream0)
        buf352 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf353 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_146, mul_93, features_37_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf353, buf351, 5376, 16, grid=grid(5376), stream=stream0)
        buf354 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_540, reinterpret_tensor(buf353, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_539, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf354)
        del primals_540
        buf355 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_148, mul_94], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf354, buf355, 224, grid=grid(224), stream=stream0)
        buf356 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_542, buf355, reinterpret_tensor(primals_541, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf356)
        del primals_542
        buf357 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_146, mul_93, mul_95], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf357, buf356, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_7], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_543, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf359 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_8, add_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf346, buf358, primals_544, primals_545, primals_546, primals_547, buf359, 14336, grid=grid(14336), stream=stream0)
        del primals_547
        # Topologically Sorted Source Nodes: [features_38_conv_0], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_548, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf361 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf362 = buf361; del buf361  # reuse
        # Topologically Sorted Source Nodes: [features_38_conv_1, sigmoid_151, mul_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf362, buf360, primals_549, primals_550, primals_551, primals_552, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_3], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_553, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf363, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf364 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf363, primals_554, primals_555, primals_556, primals_557, buf364, 86016, grid=grid(86016), stream=stream0)
        buf365 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf366 = buf365; del buf365  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_153, mul_97, features_38_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf366, buf364, 5376, 16, grid=grid(5376), stream=stream0)
        buf367 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_559, reinterpret_tensor(buf366, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_558, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf367)
        del primals_559
        buf368 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_155, mul_98], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf367, buf368, 224, grid=grid(224), stream=stream0)
        buf369 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_561, buf368, reinterpret_tensor(primals_560, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf369)
        del primals_561
        buf370 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_153, mul_97, mul_99], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf370, buf369, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_7], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_562, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf372 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_8, add_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf359, buf371, primals_563, primals_564, primals_565, primals_566, buf372, 14336, grid=grid(14336), stream=stream0)
        del primals_566
        # Topologically Sorted Source Nodes: [features_39_conv_0], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, primals_567, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf374 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf375 = buf374; del buf374  # reuse
        # Topologically Sorted Source Nodes: [features_39_conv_1, sigmoid_158, mul_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf375, buf373, primals_568, primals_569, primals_570, primals_571, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_3], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_572, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf376, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf377 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf376, primals_573, primals_574, primals_575, primals_576, buf377, 86016, grid=grid(86016), stream=stream0)
        buf378 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf379 = buf378; del buf378  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_160, mul_101, features_39_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf379, buf377, 5376, 16, grid=grid(5376), stream=stream0)
        buf380 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_578, reinterpret_tensor(buf379, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_577, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf380)
        del primals_578
        buf381 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_162, mul_102], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf380, buf381, 224, grid=grid(224), stream=stream0)
        buf382 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_580, buf381, reinterpret_tensor(primals_579, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf382)
        del primals_580
        buf383 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_160, mul_101, mul_103], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf383, buf382, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_7], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf383, primals_581, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf385 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_8, add_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf372, buf384, primals_582, primals_583, primals_584, primals_585, buf385, 14336, grid=grid(14336), stream=stream0)
        del primals_585
        # Topologically Sorted Source Nodes: [features_40_conv_0], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_586, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf387 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf388 = buf387; del buf387  # reuse
        # Topologically Sorted Source Nodes: [features_40_conv_1, sigmoid_165, mul_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf388, buf386, primals_587, primals_588, primals_589, primals_590, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_3], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_591, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf389, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf390 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf389, primals_592, primals_593, primals_594, primals_595, buf390, 86016, grid=grid(86016), stream=stream0)
        buf391 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf392 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_167, mul_105, features_40_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf392, buf390, 5376, 16, grid=grid(5376), stream=stream0)
        buf393 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_597, reinterpret_tensor(buf392, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_596, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf393)
        del primals_597
        buf394 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_169, mul_106], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf393, buf394, 224, grid=grid(224), stream=stream0)
        buf395 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_599, buf394, reinterpret_tensor(primals_598, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf395)
        del primals_599
        buf396 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_167, mul_105, mul_107], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf396, buf395, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_7], Original ATen: [aten.convolution]
        buf397 = extern_kernels.convolution(buf396, primals_600, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf397, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf398 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_8, add_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf385, buf397, primals_601, primals_602, primals_603, primals_604, buf398, 14336, grid=grid(14336), stream=stream0)
        del primals_604
        # Topologically Sorted Source Nodes: [features_41_conv_0], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_605, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf400 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf401 = buf400; del buf400  # reuse
        # Topologically Sorted Source Nodes: [features_41_conv_1, sigmoid_172, mul_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf401, buf399, primals_606, primals_607, primals_608, primals_609, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_41_conv_3], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_610, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf402, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf403 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf402, primals_611, primals_612, primals_613, primals_614, buf403, 86016, grid=grid(86016), stream=stream0)
        buf404 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf405 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_174, mul_109, features_41_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf405, buf403, 5376, 16, grid=grid(5376), stream=stream0)
        buf406 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_616, reinterpret_tensor(buf405, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_615, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf406)
        del primals_616
        buf407 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_176, mul_110], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf406, buf407, 224, grid=grid(224), stream=stream0)
        buf408 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_618, buf407, reinterpret_tensor(primals_617, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf408)
        del primals_618
        buf409 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_174, mul_109, mul_111], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf409, buf408, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_41_conv_7], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, primals_619, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf411 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_8, add_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf398, buf410, primals_620, primals_621, primals_622, primals_623, buf411, 14336, grid=grid(14336), stream=stream0)
        del primals_623
        # Topologically Sorted Source Nodes: [features_42_conv_0], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_624, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf412, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf413 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf414 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [features_42_conv_1, sigmoid_179, mul_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf414, buf412, primals_625, primals_626, primals_627, primals_628, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_42_conv_3], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf414, primals_629, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf415, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf416 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf415, primals_630, primals_631, primals_632, primals_633, buf416, 86016, grid=grid(86016), stream=stream0)
        buf417 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf418 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_181, mul_113, features_42_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf418, buf416, 5376, 16, grid=grid(5376), stream=stream0)
        buf419 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_635, reinterpret_tensor(buf418, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_634, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf419)
        del primals_635
        buf420 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_183, mul_114], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf419, buf420, 224, grid=grid(224), stream=stream0)
        buf421 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_637, buf420, reinterpret_tensor(primals_636, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf421)
        del primals_637
        buf422 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_181, mul_113, mul_115], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf422, buf421, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_42_conv_7], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf422, primals_638, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf423, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf424 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_8, add_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf411, buf423, primals_639, primals_640, primals_641, primals_642, buf424, 14336, grid=grid(14336), stream=stream0)
        del primals_642
        # Topologically Sorted Source Nodes: [features_43_conv_0], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_643, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf426 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf427 = buf426; del buf426  # reuse
        # Topologically Sorted Source Nodes: [features_43_conv_1, sigmoid_186, mul_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf427, buf425, primals_644, primals_645, primals_646, primals_647, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_43_conv_3], Original ATen: [aten.convolution]
        buf428 = extern_kernels.convolution(buf427, primals_648, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf428, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf429 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf428, primals_649, primals_650, primals_651, primals_652, buf429, 86016, grid=grid(86016), stream=stream0)
        buf430 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf431 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_188, mul_117, features_43_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf431, buf429, 5376, 16, grid=grid(5376), stream=stream0)
        buf432 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_654, reinterpret_tensor(buf431, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_653, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf432)
        del primals_654
        buf433 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_190, mul_118], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf432, buf433, 224, grid=grid(224), stream=stream0)
        buf434 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_656, buf433, reinterpret_tensor(primals_655, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf434)
        del primals_656
        buf435 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_188, mul_117, mul_119], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf435, buf434, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_43_conv_7], Original ATen: [aten.convolution]
        buf436 = extern_kernels.convolution(buf435, primals_657, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf436, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf437 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_8, add_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf424, buf436, primals_658, primals_659, primals_660, primals_661, buf437, 14336, grid=grid(14336), stream=stream0)
        del primals_661
        # Topologically Sorted Source Nodes: [features_44_conv_0], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_662, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf439 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf440 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [features_44_conv_1, sigmoid_193, mul_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf440, buf438, primals_663, primals_664, primals_665, primals_666, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_44_conv_3], Original ATen: [aten.convolution]
        buf441 = extern_kernels.convolution(buf440, primals_667, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf441, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf442 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf441, primals_668, primals_669, primals_670, primals_671, buf442, 86016, grid=grid(86016), stream=stream0)
        buf443 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf444 = buf443; del buf443  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_195, mul_121, features_44_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf444, buf442, 5376, 16, grid=grid(5376), stream=stream0)
        buf445 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_673, reinterpret_tensor(buf444, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_672, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf445)
        del primals_673
        buf446 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_197, mul_122], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf445, buf446, 224, grid=grid(224), stream=stream0)
        buf447 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_675, buf446, reinterpret_tensor(primals_674, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf447)
        del primals_675
        buf448 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_195, mul_121, mul_123], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf448, buf447, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_44_conv_7], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_676, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf450 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_8, add_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf437, buf449, primals_677, primals_678, primals_679, primals_680, buf450, 14336, grid=grid(14336), stream=stream0)
        del primals_680
        # Topologically Sorted Source Nodes: [features_45_conv_0], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, primals_681, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf452 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf453 = buf452; del buf452  # reuse
        # Topologically Sorted Source Nodes: [features_45_conv_1, sigmoid_200, mul_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf453, buf451, primals_682, primals_683, primals_684, primals_685, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_45_conv_3], Original ATen: [aten.convolution]
        buf454 = extern_kernels.convolution(buf453, primals_686, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf454, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf455 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf454, primals_687, primals_688, primals_689, primals_690, buf455, 86016, grid=grid(86016), stream=stream0)
        buf456 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf457 = buf456; del buf456  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_202, mul_125, features_45_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf457, buf455, 5376, 16, grid=grid(5376), stream=stream0)
        buf458 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_692, reinterpret_tensor(buf457, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_691, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf458)
        del primals_692
        buf459 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_204, mul_126], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf458, buf459, 224, grid=grid(224), stream=stream0)
        buf460 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_694, buf459, reinterpret_tensor(primals_693, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf460)
        del primals_694
        buf461 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_202, mul_125, mul_127], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf461, buf460, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_45_conv_7], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_695, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf463 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_8, add_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf450, buf462, primals_696, primals_697, primals_698, primals_699, buf463, 14336, grid=grid(14336), stream=stream0)
        del primals_699
        # Topologically Sorted Source Nodes: [features_46_conv_0], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_700, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf464, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf465 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf466 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [features_46_conv_1, sigmoid_207, mul_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf466, buf464, primals_701, primals_702, primals_703, primals_704, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_46_conv_3], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_705, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf467, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf468 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf467, primals_706, primals_707, primals_708, primals_709, buf468, 86016, grid=grid(86016), stream=stream0)
        buf469 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf470 = buf469; del buf469  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_209, mul_129, features_46_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf470, buf468, 5376, 16, grid=grid(5376), stream=stream0)
        buf471 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_711, reinterpret_tensor(buf470, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_710, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf471)
        del primals_711
        buf472 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_211, mul_130], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf471, buf472, 224, grid=grid(224), stream=stream0)
        buf473 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_713, buf472, reinterpret_tensor(primals_712, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf473)
        del primals_713
        buf474 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_209, mul_129, mul_131], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf474, buf473, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_46_conv_7], Original ATen: [aten.convolution]
        buf475 = extern_kernels.convolution(buf474, primals_714, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf475, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf476 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_8, add_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf463, buf475, primals_715, primals_716, primals_717, primals_718, buf476, 14336, grid=grid(14336), stream=stream0)
        del primals_718
        # Topologically Sorted Source Nodes: [features_47_conv_0], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_719, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf478 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf479 = buf478; del buf478  # reuse
        # Topologically Sorted Source Nodes: [features_47_conv_1, sigmoid_214, mul_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf479, buf477, primals_720, primals_721, primals_722, primals_723, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_47_conv_3], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, primals_724, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf480, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf481 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf480, primals_725, primals_726, primals_727, primals_728, buf481, 86016, grid=grid(86016), stream=stream0)
        buf482 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        buf483 = buf482; del buf482  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_216, mul_133, features_47_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf483, buf481, 5376, 16, grid=grid(5376), stream=stream0)
        buf484 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_730, reinterpret_tensor(buf483, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_729, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf484)
        del primals_730
        buf485 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_218, mul_134], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf484, buf485, 224, grid=grid(224), stream=stream0)
        buf486 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_732, buf485, reinterpret_tensor(primals_731, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf486)
        del primals_732
        buf487 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_216, mul_133, mul_135], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf487, buf486, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_47_conv_7], Original ATen: [aten.convolution]
        buf488 = extern_kernels.convolution(buf487, primals_733, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (4, 224, 4, 4), (3584, 1, 896, 224))
        buf489 = empty_strided_cuda((4, 224, 4, 4), (3584, 1, 896, 224), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_8, add_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf476, buf488, primals_734, primals_735, primals_736, primals_737, buf489, 14336, grid=grid(14336), stream=stream0)
        del primals_737
        # Topologically Sorted Source Nodes: [features_48_conv_0], Original ATen: [aten.convolution]
        buf490 = extern_kernels.convolution(buf489, primals_738, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 1344, 4, 4), (21504, 1, 5376, 1344))
        buf491 = empty_strided_cuda((4, 1344, 4, 4), (21504, 1, 5376, 1344), torch.float32)
        buf492 = buf491; del buf491  # reuse
        # Topologically Sorted Source Nodes: [features_48_conv_1, sigmoid_221, mul_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf492, buf490, primals_739, primals_740, primals_741, primals_742, 86016, grid=grid(86016), stream=stream0)
        # Topologically Sorted Source Nodes: [features_48_conv_3], Original ATen: [aten.convolution]
        buf493 = extern_kernels.convolution(buf492, primals_743, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1344, bias=None)
        assert_size_stride(buf493, (4, 1344, 2, 2), (5376, 1, 2688, 1344))
        buf494 = empty_strided_cuda((4, 1344, 2, 2), (5376, 1, 2688, 1344), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf493, primals_744, primals_745, primals_746, primals_747, buf494, 21504, grid=grid(21504), stream=stream0)
        buf495 = empty_strided_cuda((4, 1344, 1, 1), (1344, 1, 5376, 5376), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_223, mul_137, features_48_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_42.run(buf494, buf495, 5376, grid=grid(5376), stream=stream0)
        buf496 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_749, reinterpret_tensor(buf495, (4, 1344), (1344, 1), 0), reinterpret_tensor(primals_748, (1344, 56), (1, 1344), 0), alpha=1, beta=1, out=buf496)
        del primals_749
        buf497 = empty_strided_cuda((4, 56), (56, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_225, mul_138], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf496, buf497, 224, grid=grid(224), stream=stream0)
        buf498 = empty_strided_cuda((4, 1344), (1344, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_751, buf497, reinterpret_tensor(primals_750, (56, 1344), (1, 56), 0), alpha=1, beta=1, out=buf498)
        del primals_751
        buf499 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_223, mul_137, mul_139], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_43.run(buf499, buf498, 21504, grid=grid(21504), stream=stream0)
        # Topologically Sorted Source Nodes: [features_48_conv_7], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, primals_752, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf501 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf500, primals_753, primals_754, primals_755, primals_756, buf501, 6144, grid=grid(6144), stream=stream0)
        del primals_756
        # Topologically Sorted Source Nodes: [features_49_conv_0], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_757, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf503 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf504 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [features_49_conv_1, sigmoid_228, mul_140], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf504, buf502, primals_758, primals_759, primals_760, primals_761, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_49_conv_3], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, primals_762, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf505, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf506 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf505, primals_763, primals_764, primals_765, primals_766, buf506, 36864, grid=grid(36864), stream=stream0)
        buf507 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_230, mul_141, features_49_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf506, buf507, 9216, grid=grid(9216), stream=stream0)
        buf508 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_768, reinterpret_tensor(buf507, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_767, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf508)
        del primals_768
        buf509 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_232, mul_142], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf508, buf509, 384, grid=grid(384), stream=stream0)
        buf510 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_770, buf509, reinterpret_tensor(primals_769, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf510)
        del primals_770
        buf511 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_230, mul_141, mul_143], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf511, buf510, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_49_conv_7], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_771, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf513 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_8, add_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf501, buf512, primals_772, primals_773, primals_774, primals_775, buf513, 6144, grid=grid(6144), stream=stream0)
        del primals_775
        # Topologically Sorted Source Nodes: [features_50_conv_0], Original ATen: [aten.convolution]
        buf514 = extern_kernels.convolution(buf513, primals_776, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf515 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf516 = buf515; del buf515  # reuse
        # Topologically Sorted Source Nodes: [features_50_conv_1, sigmoid_235, mul_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf516, buf514, primals_777, primals_778, primals_779, primals_780, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_50_conv_3], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_781, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf517, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf518 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf517, primals_782, primals_783, primals_784, primals_785, buf518, 36864, grid=grid(36864), stream=stream0)
        buf519 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_237, mul_145, features_50_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf518, buf519, 9216, grid=grid(9216), stream=stream0)
        buf520 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_787, reinterpret_tensor(buf519, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_786, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf520)
        del primals_787
        buf521 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_239, mul_146], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf520, buf521, 384, grid=grid(384), stream=stream0)
        buf522 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_789, buf521, reinterpret_tensor(primals_788, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf522)
        del primals_789
        buf523 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_237, mul_145, mul_147], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf523, buf522, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_50_conv_7], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, primals_790, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf525 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_8, add_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf513, buf524, primals_791, primals_792, primals_793, primals_794, buf525, 6144, grid=grid(6144), stream=stream0)
        del primals_794
        # Topologically Sorted Source Nodes: [features_51_conv_0], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_795, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf527 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf528 = buf527; del buf527  # reuse
        # Topologically Sorted Source Nodes: [features_51_conv_1, sigmoid_242, mul_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf528, buf526, primals_796, primals_797, primals_798, primals_799, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_51_conv_3], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf528, primals_800, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf529, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf530 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf529, primals_801, primals_802, primals_803, primals_804, buf530, 36864, grid=grid(36864), stream=stream0)
        buf531 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_244, mul_149, features_51_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf530, buf531, 9216, grid=grid(9216), stream=stream0)
        buf532 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_806, reinterpret_tensor(buf531, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_805, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf532)
        del primals_806
        buf533 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_246, mul_150], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf532, buf533, 384, grid=grid(384), stream=stream0)
        buf534 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_808, buf533, reinterpret_tensor(primals_807, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf534)
        del primals_808
        buf535 = buf530; del buf530  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_244, mul_149, mul_151], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf535, buf534, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_51_conv_7], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_809, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf537 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_8, add_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf525, buf536, primals_810, primals_811, primals_812, primals_813, buf537, 6144, grid=grid(6144), stream=stream0)
        del primals_813
        # Topologically Sorted Source Nodes: [features_52_conv_0], Original ATen: [aten.convolution]
        buf538 = extern_kernels.convolution(buf537, primals_814, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf538, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf539 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf540 = buf539; del buf539  # reuse
        # Topologically Sorted Source Nodes: [features_52_conv_1, sigmoid_249, mul_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf540, buf538, primals_815, primals_816, primals_817, primals_818, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_52_conv_3], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, primals_819, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf541, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf542 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf541, primals_820, primals_821, primals_822, primals_823, buf542, 36864, grid=grid(36864), stream=stream0)
        buf543 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_251, mul_153, features_52_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf542, buf543, 9216, grid=grid(9216), stream=stream0)
        buf544 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_825, reinterpret_tensor(buf543, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_824, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf544)
        del primals_825
        buf545 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_253, mul_154], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf544, buf545, 384, grid=grid(384), stream=stream0)
        buf546 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_827, buf545, reinterpret_tensor(primals_826, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf546)
        del primals_827
        buf547 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_251, mul_153, mul_155], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf547, buf546, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_52_conv_7], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, primals_828, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf549 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_8, add_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf537, buf548, primals_829, primals_830, primals_831, primals_832, buf549, 6144, grid=grid(6144), stream=stream0)
        del primals_832
        # Topologically Sorted Source Nodes: [features_53_conv_0], Original ATen: [aten.convolution]
        buf550 = extern_kernels.convolution(buf549, primals_833, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf550, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf551 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf552 = buf551; del buf551  # reuse
        # Topologically Sorted Source Nodes: [features_53_conv_1, sigmoid_256, mul_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf552, buf550, primals_834, primals_835, primals_836, primals_837, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_53_conv_3], Original ATen: [aten.convolution]
        buf553 = extern_kernels.convolution(buf552, primals_838, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf553, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf554 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf553, primals_839, primals_840, primals_841, primals_842, buf554, 36864, grid=grid(36864), stream=stream0)
        buf555 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_258, mul_157, features_53_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf554, buf555, 9216, grid=grid(9216), stream=stream0)
        buf556 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_844, reinterpret_tensor(buf555, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_843, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf556)
        del primals_844
        buf557 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_260, mul_158], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf556, buf557, 384, grid=grid(384), stream=stream0)
        buf558 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_846, buf557, reinterpret_tensor(primals_845, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf558)
        del primals_846
        buf559 = buf554; del buf554  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_258, mul_157, mul_159], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf559, buf558, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_53_conv_7], Original ATen: [aten.convolution]
        buf560 = extern_kernels.convolution(buf559, primals_847, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf560, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf561 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_8, add_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf549, buf560, primals_848, primals_849, primals_850, primals_851, buf561, 6144, grid=grid(6144), stream=stream0)
        del primals_851
        # Topologically Sorted Source Nodes: [features_54_conv_0], Original ATen: [aten.convolution]
        buf562 = extern_kernels.convolution(buf561, primals_852, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf562, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf563 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf564 = buf563; del buf563  # reuse
        # Topologically Sorted Source Nodes: [features_54_conv_1, sigmoid_263, mul_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf564, buf562, primals_853, primals_854, primals_855, primals_856, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_54_conv_3], Original ATen: [aten.convolution]
        buf565 = extern_kernels.convolution(buf564, primals_857, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf565, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf566 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf565, primals_858, primals_859, primals_860, primals_861, buf566, 36864, grid=grid(36864), stream=stream0)
        buf567 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_265, mul_161, features_54_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf566, buf567, 9216, grid=grid(9216), stream=stream0)
        buf568 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_863, reinterpret_tensor(buf567, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_862, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf568)
        del primals_863
        buf569 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_267, mul_162], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf568, buf569, 384, grid=grid(384), stream=stream0)
        buf570 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_865, buf569, reinterpret_tensor(primals_864, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf570)
        del primals_865
        buf571 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_265, mul_161, mul_163], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf571, buf570, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_54_conv_7], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_866, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf573 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_8, add_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf561, buf572, primals_867, primals_868, primals_869, primals_870, buf573, 6144, grid=grid(6144), stream=stream0)
        del primals_870
        # Topologically Sorted Source Nodes: [features_55_conv_0], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_871, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf575 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf576 = buf575; del buf575  # reuse
        # Topologically Sorted Source Nodes: [features_55_conv_1, sigmoid_270, mul_164], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf576, buf574, primals_872, primals_873, primals_874, primals_875, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_55_conv_3], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(buf576, primals_876, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf577, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf578 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf577, primals_877, primals_878, primals_879, primals_880, buf578, 36864, grid=grid(36864), stream=stream0)
        buf579 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_272, mul_165, features_55_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf578, buf579, 9216, grid=grid(9216), stream=stream0)
        buf580 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_882, reinterpret_tensor(buf579, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_881, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf580)
        del primals_882
        buf581 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_274, mul_166], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf580, buf581, 384, grid=grid(384), stream=stream0)
        buf582 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_884, buf581, reinterpret_tensor(primals_883, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf582)
        del primals_884
        buf583 = buf578; del buf578  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_272, mul_165, mul_167], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf583, buf582, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_55_conv_7], Original ATen: [aten.convolution]
        buf584 = extern_kernels.convolution(buf583, primals_885, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf584, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf585 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_8, add_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf573, buf584, primals_886, primals_887, primals_888, primals_889, buf585, 6144, grid=grid(6144), stream=stream0)
        del primals_889
        # Topologically Sorted Source Nodes: [features_56_conv_0], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_890, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf587 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf588 = buf587; del buf587  # reuse
        # Topologically Sorted Source Nodes: [features_56_conv_1, sigmoid_277, mul_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf588, buf586, primals_891, primals_892, primals_893, primals_894, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_56_conv_3], Original ATen: [aten.convolution]
        buf589 = extern_kernels.convolution(buf588, primals_895, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf589, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf590 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf589, primals_896, primals_897, primals_898, primals_899, buf590, 36864, grid=grid(36864), stream=stream0)
        buf591 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_279, mul_169, features_56_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf590, buf591, 9216, grid=grid(9216), stream=stream0)
        buf592 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_901, reinterpret_tensor(buf591, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_900, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf592)
        del primals_901
        buf593 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_281, mul_170], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf592, buf593, 384, grid=grid(384), stream=stream0)
        buf594 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_903, buf593, reinterpret_tensor(primals_902, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf594)
        del primals_903
        buf595 = buf590; del buf590  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_279, mul_169, mul_171], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf595, buf594, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_56_conv_7], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_904, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf597 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_8, add_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf585, buf596, primals_905, primals_906, primals_907, primals_908, buf597, 6144, grid=grid(6144), stream=stream0)
        del primals_908
        # Topologically Sorted Source Nodes: [features_57_conv_0], Original ATen: [aten.convolution]
        buf598 = extern_kernels.convolution(buf597, primals_909, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf599 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf600 = buf599; del buf599  # reuse
        # Topologically Sorted Source Nodes: [features_57_conv_1, sigmoid_284, mul_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf600, buf598, primals_910, primals_911, primals_912, primals_913, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_57_conv_3], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, primals_914, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf601, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf602 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf601, primals_915, primals_916, primals_917, primals_918, buf602, 36864, grid=grid(36864), stream=stream0)
        buf603 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_286, mul_173, features_57_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf602, buf603, 9216, grid=grid(9216), stream=stream0)
        buf604 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_920, reinterpret_tensor(buf603, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_919, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf604)
        del primals_920
        buf605 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_288, mul_174], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf604, buf605, 384, grid=grid(384), stream=stream0)
        buf606 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_922, buf605, reinterpret_tensor(primals_921, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf606)
        del primals_922
        buf607 = buf602; del buf602  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_286, mul_173, mul_175], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf607, buf606, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_57_conv_7], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(buf607, primals_923, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf609 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_8, add_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf597, buf608, primals_924, primals_925, primals_926, primals_927, buf609, 6144, grid=grid(6144), stream=stream0)
        del primals_927
        # Topologically Sorted Source Nodes: [features_58_conv_0], Original ATen: [aten.convolution]
        buf610 = extern_kernels.convolution(buf609, primals_928, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf610, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf611 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf612 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [features_58_conv_1, sigmoid_291, mul_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf612, buf610, primals_929, primals_930, primals_931, primals_932, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_58_conv_3], Original ATen: [aten.convolution]
        buf613 = extern_kernels.convolution(buf612, primals_933, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf613, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf614 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf613, primals_934, primals_935, primals_936, primals_937, buf614, 36864, grid=grid(36864), stream=stream0)
        buf615 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_293, mul_177, features_58_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf614, buf615, 9216, grid=grid(9216), stream=stream0)
        buf616 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_939, reinterpret_tensor(buf615, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_938, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf616)
        del primals_939
        buf617 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_295, mul_178], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf616, buf617, 384, grid=grid(384), stream=stream0)
        buf618 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_941, buf617, reinterpret_tensor(primals_940, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf618)
        del primals_941
        buf619 = buf614; del buf614  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_293, mul_177, mul_179], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf619, buf618, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_58_conv_7], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_942, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf620, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf621 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_8, add_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf609, buf620, primals_943, primals_944, primals_945, primals_946, buf621, 6144, grid=grid(6144), stream=stream0)
        del primals_946
        # Topologically Sorted Source Nodes: [features_59_conv_0], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf621, primals_947, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf622, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf623 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf624 = buf623; del buf623  # reuse
        # Topologically Sorted Source Nodes: [features_59_conv_1, sigmoid_298, mul_180], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf624, buf622, primals_948, primals_949, primals_950, primals_951, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_59_conv_3], Original ATen: [aten.convolution]
        buf625 = extern_kernels.convolution(buf624, primals_952, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf625, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf626 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf625, primals_953, primals_954, primals_955, primals_956, buf626, 36864, grid=grid(36864), stream=stream0)
        buf627 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_300, mul_181, features_59_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf626, buf627, 9216, grid=grid(9216), stream=stream0)
        buf628 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_958, reinterpret_tensor(buf627, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_957, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf628)
        del primals_958
        buf629 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_302, mul_182], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf628, buf629, 384, grid=grid(384), stream=stream0)
        buf630 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_960, buf629, reinterpret_tensor(primals_959, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf630)
        del primals_960
        buf631 = buf626; del buf626  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_300, mul_181, mul_183], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf631, buf630, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_59_conv_7], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_961, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf633 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_8, add_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf621, buf632, primals_962, primals_963, primals_964, primals_965, buf633, 6144, grid=grid(6144), stream=stream0)
        del primals_965
        # Topologically Sorted Source Nodes: [features_60_conv_0], Original ATen: [aten.convolution]
        buf634 = extern_kernels.convolution(buf633, primals_966, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf634, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf635 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf636 = buf635; del buf635  # reuse
        # Topologically Sorted Source Nodes: [features_60_conv_1, sigmoid_305, mul_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf636, buf634, primals_967, primals_968, primals_969, primals_970, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_60_conv_3], Original ATen: [aten.convolution]
        buf637 = extern_kernels.convolution(buf636, primals_971, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf637, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf638 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf637, primals_972, primals_973, primals_974, primals_975, buf638, 36864, grid=grid(36864), stream=stream0)
        buf639 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_307, mul_185, features_60_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf638, buf639, 9216, grid=grid(9216), stream=stream0)
        buf640 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_977, reinterpret_tensor(buf639, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_976, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf640)
        del primals_977
        buf641 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_309, mul_186], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf640, buf641, 384, grid=grid(384), stream=stream0)
        buf642 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_979, buf641, reinterpret_tensor(primals_978, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf642)
        del primals_979
        buf643 = buf638; del buf638  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_307, mul_185, mul_187], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf643, buf642, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_60_conv_7], Original ATen: [aten.convolution]
        buf644 = extern_kernels.convolution(buf643, primals_980, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf644, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf645 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_8, add_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf633, buf644, primals_981, primals_982, primals_983, primals_984, buf645, 6144, grid=grid(6144), stream=stream0)
        del primals_984
        # Topologically Sorted Source Nodes: [features_61_conv_0], Original ATen: [aten.convolution]
        buf646 = extern_kernels.convolution(buf645, primals_985, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf647 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf648 = buf647; del buf647  # reuse
        # Topologically Sorted Source Nodes: [features_61_conv_1, sigmoid_312, mul_188], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf648, buf646, primals_986, primals_987, primals_988, primals_989, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_61_conv_3], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf648, primals_990, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf649, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf650 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf649, primals_991, primals_992, primals_993, primals_994, buf650, 36864, grid=grid(36864), stream=stream0)
        buf651 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_314, mul_189, features_61_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf650, buf651, 9216, grid=grid(9216), stream=stream0)
        buf652 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_996, reinterpret_tensor(buf651, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_995, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf652)
        del primals_996
        buf653 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_316, mul_190], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf652, buf653, 384, grid=grid(384), stream=stream0)
        buf654 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_998, buf653, reinterpret_tensor(primals_997, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf654)
        del primals_998
        buf655 = buf650; del buf650  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_314, mul_189, mul_191], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf655, buf654, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_61_conv_7], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_999, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf657 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_8, add_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf645, buf656, primals_1000, primals_1001, primals_1002, primals_1003, buf657, 6144, grid=grid(6144), stream=stream0)
        del primals_1003
        # Topologically Sorted Source Nodes: [features_62_conv_0], Original ATen: [aten.convolution]
        buf658 = extern_kernels.convolution(buf657, primals_1004, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf658, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf659 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf660 = buf659; del buf659  # reuse
        # Topologically Sorted Source Nodes: [features_62_conv_1, sigmoid_319, mul_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf660, buf658, primals_1005, primals_1006, primals_1007, primals_1008, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_62_conv_3], Original ATen: [aten.convolution]
        buf661 = extern_kernels.convolution(buf660, primals_1009, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf661, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf662 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf661, primals_1010, primals_1011, primals_1012, primals_1013, buf662, 36864, grid=grid(36864), stream=stream0)
        buf663 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_321, mul_193, features_62_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf662, buf663, 9216, grid=grid(9216), stream=stream0)
        buf664 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1015, reinterpret_tensor(buf663, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1014, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf664)
        del primals_1015
        buf665 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_323, mul_194], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf664, buf665, 384, grid=grid(384), stream=stream0)
        buf666 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1017, buf665, reinterpret_tensor(primals_1016, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf666)
        del primals_1017
        buf667 = buf662; del buf662  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_321, mul_193, mul_195], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf667, buf666, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_62_conv_7], Original ATen: [aten.convolution]
        buf668 = extern_kernels.convolution(buf667, primals_1018, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf669 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_8, add_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf657, buf668, primals_1019, primals_1020, primals_1021, primals_1022, buf669, 6144, grid=grid(6144), stream=stream0)
        del primals_1022
        # Topologically Sorted Source Nodes: [features_63_conv_0], Original ATen: [aten.convolution]
        buf670 = extern_kernels.convolution(buf669, primals_1023, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf670, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf671 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf672 = buf671; del buf671  # reuse
        # Topologically Sorted Source Nodes: [features_63_conv_1, sigmoid_326, mul_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf672, buf670, primals_1024, primals_1025, primals_1026, primals_1027, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_63_conv_3], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, primals_1028, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf673, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf674 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf673, primals_1029, primals_1030, primals_1031, primals_1032, buf674, 36864, grid=grid(36864), stream=stream0)
        buf675 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_328, mul_197, features_63_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf674, buf675, 9216, grid=grid(9216), stream=stream0)
        buf676 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1034, reinterpret_tensor(buf675, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1033, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf676)
        del primals_1034
        buf677 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_330, mul_198], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf676, buf677, 384, grid=grid(384), stream=stream0)
        buf678 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1036, buf677, reinterpret_tensor(primals_1035, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf678)
        del primals_1036
        buf679 = buf674; del buf674  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_328, mul_197, mul_199], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf679, buf678, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_63_conv_7], Original ATen: [aten.convolution]
        buf680 = extern_kernels.convolution(buf679, primals_1037, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf680, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf681 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_8, add_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf669, buf680, primals_1038, primals_1039, primals_1040, primals_1041, buf681, 6144, grid=grid(6144), stream=stream0)
        del primals_1041
        # Topologically Sorted Source Nodes: [features_64_conv_0], Original ATen: [aten.convolution]
        buf682 = extern_kernels.convolution(buf681, primals_1042, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf682, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf683 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf684 = buf683; del buf683  # reuse
        # Topologically Sorted Source Nodes: [features_64_conv_1, sigmoid_333, mul_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf684, buf682, primals_1043, primals_1044, primals_1045, primals_1046, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_64_conv_3], Original ATen: [aten.convolution]
        buf685 = extern_kernels.convolution(buf684, primals_1047, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf685, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf686 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf685, primals_1048, primals_1049, primals_1050, primals_1051, buf686, 36864, grid=grid(36864), stream=stream0)
        buf687 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_335, mul_201, features_64_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf686, buf687, 9216, grid=grid(9216), stream=stream0)
        buf688 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1053, reinterpret_tensor(buf687, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1052, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf688)
        del primals_1053
        buf689 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_337, mul_202], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf688, buf689, 384, grid=grid(384), stream=stream0)
        buf690 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1055, buf689, reinterpret_tensor(primals_1054, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf690)
        del primals_1055
        buf691 = buf686; del buf686  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_335, mul_201, mul_203], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf691, buf690, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_64_conv_7], Original ATen: [aten.convolution]
        buf692 = extern_kernels.convolution(buf691, primals_1056, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf692, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf693 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_8, add_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf681, buf692, primals_1057, primals_1058, primals_1059, primals_1060, buf693, 6144, grid=grid(6144), stream=stream0)
        del primals_1060
        # Topologically Sorted Source Nodes: [features_65_conv_0], Original ATen: [aten.convolution]
        buf694 = extern_kernels.convolution(buf693, primals_1061, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf694, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf695 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf696 = buf695; del buf695  # reuse
        # Topologically Sorted Source Nodes: [features_65_conv_1, sigmoid_340, mul_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf696, buf694, primals_1062, primals_1063, primals_1064, primals_1065, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_65_conv_3], Original ATen: [aten.convolution]
        buf697 = extern_kernels.convolution(buf696, primals_1066, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf697, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf698 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf697, primals_1067, primals_1068, primals_1069, primals_1070, buf698, 36864, grid=grid(36864), stream=stream0)
        buf699 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_342, mul_205, features_65_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf698, buf699, 9216, grid=grid(9216), stream=stream0)
        buf700 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1072, reinterpret_tensor(buf699, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1071, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf700)
        del primals_1072
        buf701 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_344, mul_206], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf700, buf701, 384, grid=grid(384), stream=stream0)
        buf702 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1074, buf701, reinterpret_tensor(primals_1073, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf702)
        del primals_1074
        buf703 = buf698; del buf698  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_342, mul_205, mul_207], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf703, buf702, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_65_conv_7], Original ATen: [aten.convolution]
        buf704 = extern_kernels.convolution(buf703, primals_1075, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf704, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf705 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_8, add_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf693, buf704, primals_1076, primals_1077, primals_1078, primals_1079, buf705, 6144, grid=grid(6144), stream=stream0)
        del primals_1079
        # Topologically Sorted Source Nodes: [features_66_conv_0], Original ATen: [aten.convolution]
        buf706 = extern_kernels.convolution(buf705, primals_1080, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf706, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf707 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf708 = buf707; del buf707  # reuse
        # Topologically Sorted Source Nodes: [features_66_conv_1, sigmoid_347, mul_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf708, buf706, primals_1081, primals_1082, primals_1083, primals_1084, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_66_conv_3], Original ATen: [aten.convolution]
        buf709 = extern_kernels.convolution(buf708, primals_1085, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf709, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf710 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf709, primals_1086, primals_1087, primals_1088, primals_1089, buf710, 36864, grid=grid(36864), stream=stream0)
        buf711 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_349, mul_209, features_66_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf710, buf711, 9216, grid=grid(9216), stream=stream0)
        buf712 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1091, reinterpret_tensor(buf711, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1090, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf712)
        del primals_1091
        buf713 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_351, mul_210], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf712, buf713, 384, grid=grid(384), stream=stream0)
        buf714 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1093, buf713, reinterpret_tensor(primals_1092, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf714)
        del primals_1093
        buf715 = buf710; del buf710  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_349, mul_209, mul_211], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf715, buf714, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_66_conv_7], Original ATen: [aten.convolution]
        buf716 = extern_kernels.convolution(buf715, primals_1094, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf716, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf717 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_8, add_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf705, buf716, primals_1095, primals_1096, primals_1097, primals_1098, buf717, 6144, grid=grid(6144), stream=stream0)
        del primals_1098
        # Topologically Sorted Source Nodes: [features_67_conv_0], Original ATen: [aten.convolution]
        buf718 = extern_kernels.convolution(buf717, primals_1099, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf718, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf719 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf720 = buf719; del buf719  # reuse
        # Topologically Sorted Source Nodes: [features_67_conv_1, sigmoid_354, mul_212], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf720, buf718, primals_1100, primals_1101, primals_1102, primals_1103, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_67_conv_3], Original ATen: [aten.convolution]
        buf721 = extern_kernels.convolution(buf720, primals_1104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf721, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf722 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf721, primals_1105, primals_1106, primals_1107, primals_1108, buf722, 36864, grid=grid(36864), stream=stream0)
        buf723 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_356, mul_213, features_67_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf722, buf723, 9216, grid=grid(9216), stream=stream0)
        buf724 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1110, reinterpret_tensor(buf723, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1109, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf724)
        del primals_1110
        buf725 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_358, mul_214], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf724, buf725, 384, grid=grid(384), stream=stream0)
        buf726 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1112, buf725, reinterpret_tensor(primals_1111, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf726)
        del primals_1112
        buf727 = buf722; del buf722  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_356, mul_213, mul_215], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf727, buf726, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_67_conv_7], Original ATen: [aten.convolution]
        buf728 = extern_kernels.convolution(buf727, primals_1113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf728, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf729 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_8, add_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf717, buf728, primals_1114, primals_1115, primals_1116, primals_1117, buf729, 6144, grid=grid(6144), stream=stream0)
        del primals_1117
        # Topologically Sorted Source Nodes: [features_68_conv_0], Original ATen: [aten.convolution]
        buf730 = extern_kernels.convolution(buf729, primals_1118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf730, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf731 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf732 = buf731; del buf731  # reuse
        # Topologically Sorted Source Nodes: [features_68_conv_1, sigmoid_361, mul_216], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf732, buf730, primals_1119, primals_1120, primals_1121, primals_1122, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_68_conv_3], Original ATen: [aten.convolution]
        buf733 = extern_kernels.convolution(buf732, primals_1123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf733, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf734 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf733, primals_1124, primals_1125, primals_1126, primals_1127, buf734, 36864, grid=grid(36864), stream=stream0)
        buf735 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_363, mul_217, features_68_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf734, buf735, 9216, grid=grid(9216), stream=stream0)
        buf736 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1129, reinterpret_tensor(buf735, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1128, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf736)
        del primals_1129
        buf737 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_365, mul_218], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf736, buf737, 384, grid=grid(384), stream=stream0)
        buf738 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1131, buf737, reinterpret_tensor(primals_1130, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf738)
        del primals_1131
        buf739 = buf734; del buf734  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_363, mul_217, mul_219], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf739, buf738, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_68_conv_7], Original ATen: [aten.convolution]
        buf740 = extern_kernels.convolution(buf739, primals_1132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf740, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf741 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_8, add_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf729, buf740, primals_1133, primals_1134, primals_1135, primals_1136, buf741, 6144, grid=grid(6144), stream=stream0)
        del primals_1136
        # Topologically Sorted Source Nodes: [features_69_conv_0], Original ATen: [aten.convolution]
        buf742 = extern_kernels.convolution(buf741, primals_1137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf742, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf743 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf744 = buf743; del buf743  # reuse
        # Topologically Sorted Source Nodes: [features_69_conv_1, sigmoid_368, mul_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf744, buf742, primals_1138, primals_1139, primals_1140, primals_1141, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_69_conv_3], Original ATen: [aten.convolution]
        buf745 = extern_kernels.convolution(buf744, primals_1142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf745, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf746 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf745, primals_1143, primals_1144, primals_1145, primals_1146, buf746, 36864, grid=grid(36864), stream=stream0)
        buf747 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_370, mul_221, features_69_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf746, buf747, 9216, grid=grid(9216), stream=stream0)
        buf748 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1148, reinterpret_tensor(buf747, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1147, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf748)
        del primals_1148
        buf749 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_372, mul_222], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf748, buf749, 384, grid=grid(384), stream=stream0)
        buf750 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1150, buf749, reinterpret_tensor(primals_1149, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf750)
        del primals_1150
        buf751 = buf746; del buf746  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_370, mul_221, mul_223], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf751, buf750, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_69_conv_7], Original ATen: [aten.convolution]
        buf752 = extern_kernels.convolution(buf751, primals_1151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf752, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf753 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_8, add_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf741, buf752, primals_1152, primals_1153, primals_1154, primals_1155, buf753, 6144, grid=grid(6144), stream=stream0)
        del primals_1155
        # Topologically Sorted Source Nodes: [features_70_conv_0], Original ATen: [aten.convolution]
        buf754 = extern_kernels.convolution(buf753, primals_1156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf754, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf755 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf756 = buf755; del buf755  # reuse
        # Topologically Sorted Source Nodes: [features_70_conv_1, sigmoid_375, mul_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf756, buf754, primals_1157, primals_1158, primals_1159, primals_1160, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_70_conv_3], Original ATen: [aten.convolution]
        buf757 = extern_kernels.convolution(buf756, primals_1161, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf757, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf758 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf757, primals_1162, primals_1163, primals_1164, primals_1165, buf758, 36864, grid=grid(36864), stream=stream0)
        buf759 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_377, mul_225, features_70_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf758, buf759, 9216, grid=grid(9216), stream=stream0)
        buf760 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1167, reinterpret_tensor(buf759, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1166, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf760)
        del primals_1167
        buf761 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_379, mul_226], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf760, buf761, 384, grid=grid(384), stream=stream0)
        buf762 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1169, buf761, reinterpret_tensor(primals_1168, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf762)
        del primals_1169
        buf763 = buf758; del buf758  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_377, mul_225, mul_227], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf763, buf762, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_70_conv_7], Original ATen: [aten.convolution]
        buf764 = extern_kernels.convolution(buf763, primals_1170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf764, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf765 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_8, add_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf753, buf764, primals_1171, primals_1172, primals_1173, primals_1174, buf765, 6144, grid=grid(6144), stream=stream0)
        del primals_1174
        # Topologically Sorted Source Nodes: [features_71_conv_0], Original ATen: [aten.convolution]
        buf766 = extern_kernels.convolution(buf765, primals_1175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf766, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf767 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf768 = buf767; del buf767  # reuse
        # Topologically Sorted Source Nodes: [features_71_conv_1, sigmoid_382, mul_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf768, buf766, primals_1176, primals_1177, primals_1178, primals_1179, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_71_conv_3], Original ATen: [aten.convolution]
        buf769 = extern_kernels.convolution(buf768, primals_1180, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf769, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf770 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf769, primals_1181, primals_1182, primals_1183, primals_1184, buf770, 36864, grid=grid(36864), stream=stream0)
        buf771 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_384, mul_229, features_71_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf770, buf771, 9216, grid=grid(9216), stream=stream0)
        buf772 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1186, reinterpret_tensor(buf771, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1185, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf772)
        del primals_1186
        buf773 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_386, mul_230], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf772, buf773, 384, grid=grid(384), stream=stream0)
        buf774 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1188, buf773, reinterpret_tensor(primals_1187, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf774)
        del primals_1188
        buf775 = buf770; del buf770  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_384, mul_229, mul_231], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf775, buf774, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_71_conv_7], Original ATen: [aten.convolution]
        buf776 = extern_kernels.convolution(buf775, primals_1189, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf776, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf777 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_8, add_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf765, buf776, primals_1190, primals_1191, primals_1192, primals_1193, buf777, 6144, grid=grid(6144), stream=stream0)
        del primals_1193
        # Topologically Sorted Source Nodes: [features_72_conv_0], Original ATen: [aten.convolution]
        buf778 = extern_kernels.convolution(buf777, primals_1194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf778, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf779 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf780 = buf779; del buf779  # reuse
        # Topologically Sorted Source Nodes: [features_72_conv_1, sigmoid_389, mul_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf780, buf778, primals_1195, primals_1196, primals_1197, primals_1198, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_72_conv_3], Original ATen: [aten.convolution]
        buf781 = extern_kernels.convolution(buf780, primals_1199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf781, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf782 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf781, primals_1200, primals_1201, primals_1202, primals_1203, buf782, 36864, grid=grid(36864), stream=stream0)
        buf783 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_391, mul_233, features_72_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf782, buf783, 9216, grid=grid(9216), stream=stream0)
        buf784 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1205, reinterpret_tensor(buf783, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1204, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf784)
        del primals_1205
        buf785 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_393, mul_234], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf784, buf785, 384, grid=grid(384), stream=stream0)
        buf786 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1207, buf785, reinterpret_tensor(primals_1206, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf786)
        del primals_1207
        buf787 = buf782; del buf782  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_391, mul_233, mul_235], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf787, buf786, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_72_conv_7], Original ATen: [aten.convolution]
        buf788 = extern_kernels.convolution(buf787, primals_1208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf788, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf789 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_8, add_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf777, buf788, primals_1209, primals_1210, primals_1211, primals_1212, buf789, 6144, grid=grid(6144), stream=stream0)
        del primals_1212
        # Topologically Sorted Source Nodes: [features_73_conv_0], Original ATen: [aten.convolution]
        buf790 = extern_kernels.convolution(buf789, primals_1213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf790, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf791 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        buf792 = buf791; del buf791  # reuse
        # Topologically Sorted Source Nodes: [features_73_conv_1, sigmoid_396, mul_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf792, buf790, primals_1214, primals_1215, primals_1216, primals_1217, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_73_conv_3], Original ATen: [aten.convolution]
        buf793 = extern_kernels.convolution(buf792, primals_1218, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2304, bias=None)
        assert_size_stride(buf793, (4, 2304, 2, 2), (9216, 1, 4608, 2304))
        buf794 = empty_strided_cuda((4, 2304, 2, 2), (9216, 1, 4608, 2304), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf793, primals_1219, primals_1220, primals_1221, primals_1222, buf794, 36864, grid=grid(36864), stream=stream0)
        buf795 = empty_strided_cuda((4, 2304, 1, 1), (2304, 1, 9216, 9216), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_398, mul_237, features_73_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf794, buf795, 9216, grid=grid(9216), stream=stream0)
        buf796 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1224, reinterpret_tensor(buf795, (4, 2304), (2304, 1), 0), reinterpret_tensor(primals_1223, (2304, 96), (1, 2304), 0), alpha=1, beta=1, out=buf796)
        del primals_1224
        buf797 = empty_strided_cuda((4, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_400, mul_238], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf796, buf797, 384, grid=grid(384), stream=stream0)
        buf798 = empty_strided_cuda((4, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1226, buf797, reinterpret_tensor(primals_1225, (96, 2304), (1, 96), 0), alpha=1, beta=1, out=buf798)
        del primals_1226
        buf799 = buf794; del buf794  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_398, mul_237, mul_239], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf799, buf798, 36864, grid=grid(36864), stream=stream0)
        # Topologically Sorted Source Nodes: [features_73_conv_7], Original ATen: [aten.convolution]
        buf800 = extern_kernels.convolution(buf799, primals_1227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf800, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf801 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf800, primals_1228, primals_1229, primals_1230, primals_1231, buf801, 10240, grid=grid(10240), stream=stream0)
        del primals_1231
        # Topologically Sorted Source Nodes: [features_74_conv_0], Original ATen: [aten.convolution]
        buf802 = extern_kernels.convolution(buf801, primals_1232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf802, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf803 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf804 = buf803; del buf803  # reuse
        # Topologically Sorted Source Nodes: [features_74_conv_1, sigmoid_403, mul_240], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf804, buf802, primals_1233, primals_1234, primals_1235, primals_1236, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_74_conv_3], Original ATen: [aten.convolution]
        buf805 = extern_kernels.convolution(buf804, primals_1237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf805, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf806 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf805, primals_1238, primals_1239, primals_1240, primals_1241, buf806, 61440, grid=grid(61440), stream=stream0)
        buf807 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_405, mul_241, features_74_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf806, buf807, 15360, grid=grid(15360), stream=stream0)
        buf808 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1243, reinterpret_tensor(buf807, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1242, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf808)
        del primals_1243
        buf809 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_407, mul_242], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf808, buf809, 640, grid=grid(640), stream=stream0)
        buf810 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1245, buf809, reinterpret_tensor(primals_1244, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf810)
        del primals_1245
        buf811 = buf806; del buf806  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_405, mul_241, mul_243], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf811, buf810, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_74_conv_7], Original ATen: [aten.convolution]
        buf812 = extern_kernels.convolution(buf811, primals_1246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf812, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf813 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_8, add_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf801, buf812, primals_1247, primals_1248, primals_1249, primals_1250, buf813, 10240, grid=grid(10240), stream=stream0)
        del primals_1250
        # Topologically Sorted Source Nodes: [features_75_conv_0], Original ATen: [aten.convolution]
        buf814 = extern_kernels.convolution(buf813, primals_1251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf814, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf815 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf816 = buf815; del buf815  # reuse
        # Topologically Sorted Source Nodes: [features_75_conv_1, sigmoid_410, mul_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf816, buf814, primals_1252, primals_1253, primals_1254, primals_1255, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_75_conv_3], Original ATen: [aten.convolution]
        buf817 = extern_kernels.convolution(buf816, primals_1256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf817, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf818 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf817, primals_1257, primals_1258, primals_1259, primals_1260, buf818, 61440, grid=grid(61440), stream=stream0)
        buf819 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_412, mul_245, features_75_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf818, buf819, 15360, grid=grid(15360), stream=stream0)
        buf820 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1262, reinterpret_tensor(buf819, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1261, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf820)
        del primals_1262
        buf821 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_414, mul_246], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf820, buf821, 640, grid=grid(640), stream=stream0)
        buf822 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1264, buf821, reinterpret_tensor(primals_1263, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf822)
        del primals_1264
        buf823 = buf818; del buf818  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_412, mul_245, mul_247], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf823, buf822, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_75_conv_7], Original ATen: [aten.convolution]
        buf824 = extern_kernels.convolution(buf823, primals_1265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf824, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf825 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_8, add_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf813, buf824, primals_1266, primals_1267, primals_1268, primals_1269, buf825, 10240, grid=grid(10240), stream=stream0)
        del primals_1269
        # Topologically Sorted Source Nodes: [features_76_conv_0], Original ATen: [aten.convolution]
        buf826 = extern_kernels.convolution(buf825, primals_1270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf826, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf827 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf828 = buf827; del buf827  # reuse
        # Topologically Sorted Source Nodes: [features_76_conv_1, sigmoid_417, mul_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf828, buf826, primals_1271, primals_1272, primals_1273, primals_1274, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_76_conv_3], Original ATen: [aten.convolution]
        buf829 = extern_kernels.convolution(buf828, primals_1275, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf829, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf830 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf829, primals_1276, primals_1277, primals_1278, primals_1279, buf830, 61440, grid=grid(61440), stream=stream0)
        buf831 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_419, mul_249, features_76_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf830, buf831, 15360, grid=grid(15360), stream=stream0)
        buf832 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1281, reinterpret_tensor(buf831, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1280, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf832)
        del primals_1281
        buf833 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_421, mul_250], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf832, buf833, 640, grid=grid(640), stream=stream0)
        buf834 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1283, buf833, reinterpret_tensor(primals_1282, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf834)
        del primals_1283
        buf835 = buf830; del buf830  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_419, mul_249, mul_251], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf835, buf834, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_76_conv_7], Original ATen: [aten.convolution]
        buf836 = extern_kernels.convolution(buf835, primals_1284, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf836, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf837 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_8, add_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf825, buf836, primals_1285, primals_1286, primals_1287, primals_1288, buf837, 10240, grid=grid(10240), stream=stream0)
        del primals_1288
        # Topologically Sorted Source Nodes: [features_77_conv_0], Original ATen: [aten.convolution]
        buf838 = extern_kernels.convolution(buf837, primals_1289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf838, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf839 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf840 = buf839; del buf839  # reuse
        # Topologically Sorted Source Nodes: [features_77_conv_1, sigmoid_424, mul_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf840, buf838, primals_1290, primals_1291, primals_1292, primals_1293, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_77_conv_3], Original ATen: [aten.convolution]
        buf841 = extern_kernels.convolution(buf840, primals_1294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf841, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf842 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf841, primals_1295, primals_1296, primals_1297, primals_1298, buf842, 61440, grid=grid(61440), stream=stream0)
        buf843 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_426, mul_253, features_77_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf842, buf843, 15360, grid=grid(15360), stream=stream0)
        buf844 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1300, reinterpret_tensor(buf843, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1299, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf844)
        del primals_1300
        buf845 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_428, mul_254], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf844, buf845, 640, grid=grid(640), stream=stream0)
        buf846 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1302, buf845, reinterpret_tensor(primals_1301, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf846)
        del primals_1302
        buf847 = buf842; del buf842  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_426, mul_253, mul_255], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf847, buf846, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_77_conv_7], Original ATen: [aten.convolution]
        buf848 = extern_kernels.convolution(buf847, primals_1303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf848, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf849 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_8, add_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf837, buf848, primals_1304, primals_1305, primals_1306, primals_1307, buf849, 10240, grid=grid(10240), stream=stream0)
        del primals_1307
        # Topologically Sorted Source Nodes: [features_78_conv_0], Original ATen: [aten.convolution]
        buf850 = extern_kernels.convolution(buf849, primals_1308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf850, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf851 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf852 = buf851; del buf851  # reuse
        # Topologically Sorted Source Nodes: [features_78_conv_1, sigmoid_431, mul_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf852, buf850, primals_1309, primals_1310, primals_1311, primals_1312, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_78_conv_3], Original ATen: [aten.convolution]
        buf853 = extern_kernels.convolution(buf852, primals_1313, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf853, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf854 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf853, primals_1314, primals_1315, primals_1316, primals_1317, buf854, 61440, grid=grid(61440), stream=stream0)
        buf855 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_433, mul_257, features_78_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf854, buf855, 15360, grid=grid(15360), stream=stream0)
        buf856 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1319, reinterpret_tensor(buf855, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1318, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf856)
        del primals_1319
        buf857 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_435, mul_258], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf856, buf857, 640, grid=grid(640), stream=stream0)
        buf858 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1321, buf857, reinterpret_tensor(primals_1320, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf858)
        del primals_1321
        buf859 = buf854; del buf854  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_433, mul_257, mul_259], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf859, buf858, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_78_conv_7], Original ATen: [aten.convolution]
        buf860 = extern_kernels.convolution(buf859, primals_1322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf860, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf861 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_8, add_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf849, buf860, primals_1323, primals_1324, primals_1325, primals_1326, buf861, 10240, grid=grid(10240), stream=stream0)
        del primals_1326
        # Topologically Sorted Source Nodes: [features_79_conv_0], Original ATen: [aten.convolution]
        buf862 = extern_kernels.convolution(buf861, primals_1327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf862, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf863 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf864 = buf863; del buf863  # reuse
        # Topologically Sorted Source Nodes: [features_79_conv_1, sigmoid_438, mul_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf864, buf862, primals_1328, primals_1329, primals_1330, primals_1331, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_79_conv_3], Original ATen: [aten.convolution]
        buf865 = extern_kernels.convolution(buf864, primals_1332, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf865, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf866 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf865, primals_1333, primals_1334, primals_1335, primals_1336, buf866, 61440, grid=grid(61440), stream=stream0)
        buf867 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_440, mul_261, features_79_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf866, buf867, 15360, grid=grid(15360), stream=stream0)
        buf868 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1338, reinterpret_tensor(buf867, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1337, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf868)
        del primals_1338
        buf869 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_442, mul_262], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf868, buf869, 640, grid=grid(640), stream=stream0)
        buf870 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1340, buf869, reinterpret_tensor(primals_1339, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf870)
        del primals_1340
        buf871 = buf866; del buf866  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_440, mul_261, mul_263], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf871, buf870, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_79_conv_7], Original ATen: [aten.convolution]
        buf872 = extern_kernels.convolution(buf871, primals_1341, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf872, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf873 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_8, add_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf861, buf872, primals_1342, primals_1343, primals_1344, primals_1345, buf873, 10240, grid=grid(10240), stream=stream0)
        del primals_1345
        # Topologically Sorted Source Nodes: [conv_0], Original ATen: [aten.convolution]
        buf874 = extern_kernels.convolution(buf873, primals_1346, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf874, (4, 1792, 2, 2), (7168, 1, 3584, 1792))
        buf875 = empty_strided_cuda((4, 1792, 2, 2), (7168, 1, 3584, 1792), torch.float32)
        # Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_58.run(buf874, primals_1347, primals_1348, primals_1349, primals_1350, buf875, 28672, grid=grid(28672), stream=stream0)
        buf876 = empty_strided_cuda((4, 1792, 1, 1), (1792, 1, 7168, 7168), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_445, mul_264, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_59.run(buf875, buf876, 7168, grid=grid(7168), stream=stream0)
        del buf875
        buf877 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [classifier], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1352, reinterpret_tensor(buf876, (4, 1792), (1792, 1), 0), reinterpret_tensor(primals_1351, (1792, 1000), (1, 1792), 0), alpha=1, beta=1, out=buf877)
        del primals_1352
    return (buf877, buf0, buf1, primals_3, primals_4, primals_5, primals_6, buf2, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, buf3, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, buf4, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, buf5, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, buf6, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, buf7, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, buf8, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, buf9, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, buf10, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, buf11, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, buf12, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, buf13, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, buf14, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, buf15, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, buf16, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, buf17, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, buf18, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, buf19, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_201, primals_202, primals_203, primals_204, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_220, primals_221, primals_222, primals_223, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_239, primals_240, primals_241, primals_242, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_258, primals_259, primals_260, primals_261, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_296, primals_297, primals_298, primals_299, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_315, primals_316, primals_317, primals_318, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_334, primals_335, primals_336, primals_337, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_353, primals_354, primals_355, primals_356, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_391, primals_392, primals_393, primals_394, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_410, primals_411, primals_412, primals_413, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_429, primals_430, primals_431, primals_432, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_448, primals_449, primals_450, primals_451, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_467, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_486, primals_487, primals_488, primals_489, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_505, primals_506, primals_507, primals_508, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_524, primals_525, primals_526, primals_527, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_543, primals_544, primals_545, primals_546, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_562, primals_563, primals_564, primals_565, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_581, primals_582, primals_583, primals_584, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_600, primals_601, primals_602, primals_603, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_619, primals_620, primals_621, primals_622, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_638, primals_639, primals_640, primals_641, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_657, primals_658, primals_659, primals_660, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_676, primals_677, primals_678, primals_679, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_695, primals_696, primals_697, primals_698, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_714, primals_715, primals_716, primals_717, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_733, primals_734, primals_735, primals_736, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_752, primals_753, primals_754, primals_755, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_771, primals_772, primals_773, primals_774, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_790, primals_791, primals_792, primals_793, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_809, primals_810, primals_811, primals_812, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_828, primals_829, primals_830, primals_831, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_847, primals_848, primals_849, primals_850, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_866, primals_867, primals_868, primals_869, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_885, primals_886, primals_887, primals_888, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_904, primals_905, primals_906, primals_907, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_923, primals_924, primals_925, primals_926, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_942, primals_943, primals_944, primals_945, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_961, primals_962, primals_963, primals_964, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_980, primals_981, primals_982, primals_983, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_999, primals_1000, primals_1001, primals_1002, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1018, primals_1019, primals_1020, primals_1021, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1037, primals_1038, primals_1039, primals_1040, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1056, primals_1057, primals_1058, primals_1059, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1075, primals_1076, primals_1077, primals_1078, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1094, primals_1095, primals_1096, primals_1097, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1113, primals_1114, primals_1115, primals_1116, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1132, primals_1133, primals_1134, primals_1135, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1151, primals_1152, primals_1153, primals_1154, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1170, primals_1171, primals_1172, primals_1173, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1189, primals_1190, primals_1191, primals_1192, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1208, primals_1209, primals_1210, primals_1211, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1227, primals_1228, primals_1229, primals_1230, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1246, primals_1247, primals_1248, primals_1249, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1265, primals_1266, primals_1267, primals_1268, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1284, primals_1285, primals_1286, primals_1287, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1303, primals_1304, primals_1305, primals_1306, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1322, primals_1323, primals_1324, primals_1325, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1341, primals_1342, primals_1343, primals_1344, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, buf20, buf22, buf23, buf25, buf26, buf27, buf28, buf30, buf31, buf32, buf33, buf35, buf36, buf37, buf38, buf40, buf41, buf42, buf43, buf45, buf46, buf47, buf48, buf50, buf51, buf52, buf53, buf55, buf56, buf57, buf58, buf60, buf61, buf62, buf63, buf65, buf66, buf67, buf68, buf70, buf71, buf72, buf73, buf75, buf76, buf77, buf78, buf80, buf81, buf82, buf83, buf85, buf86, buf87, buf88, buf90, buf91, buf92, buf93, buf95, buf96, buf97, buf98, buf100, buf101, buf102, buf103, buf105, buf106, buf107, buf108, buf110, buf111, buf112, buf113, buf115, buf116, reinterpret_tensor(buf119, (4, 384), (384, 1), 0), buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf128, buf129, reinterpret_tensor(buf132, (4, 768), (768, 1), 0), buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf141, buf142, reinterpret_tensor(buf145, (4, 768), (768, 1), 0), buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf154, buf155, reinterpret_tensor(buf158, (4, 768), (768, 1), 0), buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf167, buf168, reinterpret_tensor(buf171, (4, 768), (768, 1), 0), buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf180, buf181, reinterpret_tensor(buf184, (4, 768), (768, 1), 0), buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf193, buf194, reinterpret_tensor(buf197, (4, 768), (768, 1), 0), buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf206, buf207, reinterpret_tensor(buf210, (4, 768), (768, 1), 0), buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf219, buf220, reinterpret_tensor(buf223, (4, 768), (768, 1), 0), buf224, buf225, buf226, buf227, buf228, buf229, buf230, buf232, buf233, reinterpret_tensor(buf236, (4, 768), (768, 1), 0), buf237, buf238, buf239, buf240, buf241, buf242, buf243, buf245, buf246, reinterpret_tensor(buf249, (4, 1152), (1152, 1), 0), buf250, buf251, buf252, buf253, buf254, buf255, buf256, buf258, buf259, reinterpret_tensor(buf262, (4, 1344), (1344, 1), 0), buf263, buf264, buf265, buf266, buf267, buf268, buf269, buf271, buf272, reinterpret_tensor(buf275, (4, 1344), (1344, 1), 0), buf276, buf277, buf278, buf279, buf280, buf281, buf282, buf284, buf285, reinterpret_tensor(buf288, (4, 1344), (1344, 1), 0), buf289, buf290, buf291, buf292, buf293, buf294, buf295, buf297, buf298, reinterpret_tensor(buf301, (4, 1344), (1344, 1), 0), buf302, buf303, buf304, buf305, buf306, buf307, buf308, buf310, buf311, reinterpret_tensor(buf314, (4, 1344), (1344, 1), 0), buf315, buf316, buf317, buf318, buf319, buf320, buf321, buf323, buf324, reinterpret_tensor(buf327, (4, 1344), (1344, 1), 0), buf328, buf329, buf330, buf331, buf332, buf333, buf334, buf336, buf337, reinterpret_tensor(buf340, (4, 1344), (1344, 1), 0), buf341, buf342, buf343, buf344, buf345, buf346, buf347, buf349, buf350, reinterpret_tensor(buf353, (4, 1344), (1344, 1), 0), buf354, buf355, buf356, buf357, buf358, buf359, buf360, buf362, buf363, reinterpret_tensor(buf366, (4, 1344), (1344, 1), 0), buf367, buf368, buf369, buf370, buf371, buf372, buf373, buf375, buf376, reinterpret_tensor(buf379, (4, 1344), (1344, 1), 0), buf380, buf381, buf382, buf383, buf384, buf385, buf386, buf388, buf389, reinterpret_tensor(buf392, (4, 1344), (1344, 1), 0), buf393, buf394, buf395, buf396, buf397, buf398, buf399, buf401, buf402, reinterpret_tensor(buf405, (4, 1344), (1344, 1), 0), buf406, buf407, buf408, buf409, buf410, buf411, buf412, buf414, buf415, reinterpret_tensor(buf418, (4, 1344), (1344, 1), 0), buf419, buf420, buf421, buf422, buf423, buf424, buf425, buf427, buf428, reinterpret_tensor(buf431, (4, 1344), (1344, 1), 0), buf432, buf433, buf434, buf435, buf436, buf437, buf438, buf440, buf441, reinterpret_tensor(buf444, (4, 1344), (1344, 1), 0), buf445, buf446, buf447, buf448, buf449, buf450, buf451, buf453, buf454, reinterpret_tensor(buf457, (4, 1344), (1344, 1), 0), buf458, buf459, buf460, buf461, buf462, buf463, buf464, buf466, buf467, reinterpret_tensor(buf470, (4, 1344), (1344, 1), 0), buf471, buf472, buf473, buf474, buf475, buf476, buf477, buf479, buf480, reinterpret_tensor(buf483, (4, 1344), (1344, 1), 0), buf484, buf485, buf486, buf487, buf488, buf489, buf490, buf492, buf493, reinterpret_tensor(buf495, (4, 1344), (1344, 1), 0), buf496, buf497, buf498, buf499, buf500, buf501, buf502, buf504, buf505, reinterpret_tensor(buf507, (4, 2304), (2304, 1), 0), buf508, buf509, buf510, buf511, buf512, buf513, buf514, buf516, buf517, reinterpret_tensor(buf519, (4, 2304), (2304, 1), 0), buf520, buf521, buf522, buf523, buf524, buf525, buf526, buf528, buf529, reinterpret_tensor(buf531, (4, 2304), (2304, 1), 0), buf532, buf533, buf534, buf535, buf536, buf537, buf538, buf540, buf541, reinterpret_tensor(buf543, (4, 2304), (2304, 1), 0), buf544, buf545, buf546, buf547, buf548, buf549, buf550, buf552, buf553, reinterpret_tensor(buf555, (4, 2304), (2304, 1), 0), buf556, buf557, buf558, buf559, buf560, buf561, buf562, buf564, buf565, reinterpret_tensor(buf567, (4, 2304), (2304, 1), 0), buf568, buf569, buf570, buf571, buf572, buf573, buf574, buf576, buf577, reinterpret_tensor(buf579, (4, 2304), (2304, 1), 0), buf580, buf581, buf582, buf583, buf584, buf585, buf586, buf588, buf589, reinterpret_tensor(buf591, (4, 2304), (2304, 1), 0), buf592, buf593, buf594, buf595, buf596, buf597, buf598, buf600, buf601, reinterpret_tensor(buf603, (4, 2304), (2304, 1), 0), buf604, buf605, buf606, buf607, buf608, buf609, buf610, buf612, buf613, reinterpret_tensor(buf615, (4, 2304), (2304, 1), 0), buf616, buf617, buf618, buf619, buf620, buf621, buf622, buf624, buf625, reinterpret_tensor(buf627, (4, 2304), (2304, 1), 0), buf628, buf629, buf630, buf631, buf632, buf633, buf634, buf636, buf637, reinterpret_tensor(buf639, (4, 2304), (2304, 1), 0), buf640, buf641, buf642, buf643, buf644, buf645, buf646, buf648, buf649, reinterpret_tensor(buf651, (4, 2304), (2304, 1), 0), buf652, buf653, buf654, buf655, buf656, buf657, buf658, buf660, buf661, reinterpret_tensor(buf663, (4, 2304), (2304, 1), 0), buf664, buf665, buf666, buf667, buf668, buf669, buf670, buf672, buf673, reinterpret_tensor(buf675, (4, 2304), (2304, 1), 0), buf676, buf677, buf678, buf679, buf680, buf681, buf682, buf684, buf685, reinterpret_tensor(buf687, (4, 2304), (2304, 1), 0), buf688, buf689, buf690, buf691, buf692, buf693, buf694, buf696, buf697, reinterpret_tensor(buf699, (4, 2304), (2304, 1), 0), buf700, buf701, buf702, buf703, buf704, buf705, buf706, buf708, buf709, reinterpret_tensor(buf711, (4, 2304), (2304, 1), 0), buf712, buf713, buf714, buf715, buf716, buf717, buf718, buf720, buf721, reinterpret_tensor(buf723, (4, 2304), (2304, 1), 0), buf724, buf725, buf726, buf727, buf728, buf729, buf730, buf732, buf733, reinterpret_tensor(buf735, (4, 2304), (2304, 1), 0), buf736, buf737, buf738, buf739, buf740, buf741, buf742, buf744, buf745, reinterpret_tensor(buf747, (4, 2304), (2304, 1), 0), buf748, buf749, buf750, buf751, buf752, buf753, buf754, buf756, buf757, reinterpret_tensor(buf759, (4, 2304), (2304, 1), 0), buf760, buf761, buf762, buf763, buf764, buf765, buf766, buf768, buf769, reinterpret_tensor(buf771, (4, 2304), (2304, 1), 0), buf772, buf773, buf774, buf775, buf776, buf777, buf778, buf780, buf781, reinterpret_tensor(buf783, (4, 2304), (2304, 1), 0), buf784, buf785, buf786, buf787, buf788, buf789, buf790, buf792, buf793, reinterpret_tensor(buf795, (4, 2304), (2304, 1), 0), buf796, buf797, buf798, buf799, buf800, buf801, buf802, buf804, buf805, reinterpret_tensor(buf807, (4, 3840), (3840, 1), 0), buf808, buf809, buf810, buf811, buf812, buf813, buf814, buf816, buf817, reinterpret_tensor(buf819, (4, 3840), (3840, 1), 0), buf820, buf821, buf822, buf823, buf824, buf825, buf826, buf828, buf829, reinterpret_tensor(buf831, (4, 3840), (3840, 1), 0), buf832, buf833, buf834, buf835, buf836, buf837, buf838, buf840, buf841, reinterpret_tensor(buf843, (4, 3840), (3840, 1), 0), buf844, buf845, buf846, buf847, buf848, buf849, buf850, buf852, buf853, reinterpret_tensor(buf855, (4, 3840), (3840, 1), 0), buf856, buf857, buf858, buf859, buf860, buf861, buf862, buf864, buf865, reinterpret_tensor(buf867, (4, 3840), (3840, 1), 0), buf868, buf869, buf870, buf871, buf872, buf873, buf874, reinterpret_tensor(buf876, (4, 1792), (1792, 1), 0), primals_1351, primals_1339, primals_1337, primals_1320, primals_1318, primals_1301, primals_1299, primals_1282, primals_1280, primals_1263, primals_1261, primals_1244, primals_1242, primals_1225, primals_1223, primals_1206, primals_1204, primals_1187, primals_1185, primals_1168, primals_1166, primals_1149, primals_1147, primals_1130, primals_1128, primals_1111, primals_1109, primals_1092, primals_1090, primals_1073, primals_1071, primals_1054, primals_1052, primals_1035, primals_1033, primals_1016, primals_1014, primals_997, primals_995, primals_978, primals_976, primals_959, primals_957, primals_940, primals_938, primals_921, primals_919, primals_902, primals_900, primals_883, primals_881, primals_864, primals_862, primals_845, primals_843, primals_826, primals_824, primals_807, primals_805, primals_788, primals_786, primals_769, primals_767, primals_750, primals_748, primals_731, primals_729, primals_712, primals_710, primals_693, primals_691, primals_674, primals_672, primals_655, primals_653, primals_636, primals_634, primals_617, primals_615, primals_598, primals_596, primals_579, primals_577, primals_560, primals_558, primals_541, primals_539, primals_522, primals_520, primals_503, primals_501, primals_484, primals_482, primals_465, primals_463, primals_446, primals_444, primals_427, primals_425, primals_408, primals_406, primals_389, primals_387, primals_370, primals_368, primals_351, primals_349, primals_332, primals_330, primals_313, primals_311, primals_294, primals_292, primals_275, primals_273, primals_256, primals_254, primals_237, primals_235, primals_218, primals_216, primals_199, primals_197, )


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
    primals_12 = rand_strided((32, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_107 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((24, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((48, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((1152, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((224, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((224, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((1344, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((1344, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((56, 1344), (1344, 1), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((1344, 56), (56, 1), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((384, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_981 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_984 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_987 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_990 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_993 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_996 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_999 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1002 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1005 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1008 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1011 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1014 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1017 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1020 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1024 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1025 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1026 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1027 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1028 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1029 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1030 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1031 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1032 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1033 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1034 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1035 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1036 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1037 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1038 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1039 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1040 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1041 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1042 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1043 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1044 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1045 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1046 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1047 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1048 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1049 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1050 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1051 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1052 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1053 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1054 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1055 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1056 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1057 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1058 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1059 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1060 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1061 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1062 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1063 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1064 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1065 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1066 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1067 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1068 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1069 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1070 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1071 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1072 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1073 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1074 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1075 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1076 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1077 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1078 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1079 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1080 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1081 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1082 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1083 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1084 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1085 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1086 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1087 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1088 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1089 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1090 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1091 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1092 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1093 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1094 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1095 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1096 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1097 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1098 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1099 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1100 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1101 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1102 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1103 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1104 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1105 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1106 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1107 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1108 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1109 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1110 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1111 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1112 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1113 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1115 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1116 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1118 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1119 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1120 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1121 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1122 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1123 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1124 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1125 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1126 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1127 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1128 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1129 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1130 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1131 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1132 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1137 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1138 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1139 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1140 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1141 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1142 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1143 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1144 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1145 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1146 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1147 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1148 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1149 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1150 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1151 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1152 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1156 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1157 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1158 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1159 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1160 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1161 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1162 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1163 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1164 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1165 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1166 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1167 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1168 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1169 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1170 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1174 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1175 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1176 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1177 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1178 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1179 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1180 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1181 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1182 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1183 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1184 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1185 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1186 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1187 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1188 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1189 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1190 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1191 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1192 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1194 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1195 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1196 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1197 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1198 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1199 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1200 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1201 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1202 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1203 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1204 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1205 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1206 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1207 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1208 = rand_strided((384, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1209 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1210 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1211 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1212 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1213 = rand_strided((2304, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1214 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1215 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1216 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1217 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1218 = rand_strided((2304, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1219 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1220 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1221 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1222 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1223 = rand_strided((96, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_1224 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1225 = rand_strided((2304, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_1226 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1227 = rand_strided((640, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1228 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1229 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1230 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1231 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1232 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1233 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1234 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1235 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1236 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1237 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1238 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1239 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1240 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1241 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1242 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1243 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1244 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1245 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1246 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1247 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1248 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1249 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1250 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1251 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1252 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1253 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1254 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1255 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1256 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1257 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1258 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1259 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1260 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1261 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1262 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1263 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1264 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1265 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1266 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1267 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1268 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1269 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1270 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1271 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1272 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1273 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1274 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1275 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1276 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1277 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1278 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1279 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1280 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1281 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1282 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1283 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1284 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1285 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1286 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1287 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1288 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1289 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1290 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1291 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1292 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1293 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1294 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1295 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1296 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1297 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1298 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1299 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1300 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1301 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1302 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1303 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1304 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1305 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1306 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1307 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1308 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1309 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1310 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1311 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1312 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1313 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1314 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1315 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1316 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1317 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1318 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1319 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1320 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1321 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1322 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1323 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1324 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1325 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1326 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1327 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1328 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1329 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1330 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1331 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1332 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1333 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1334 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1335 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1336 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1337 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1338 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1339 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1340 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1341 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1342 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1343 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1344 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1345 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1346 = rand_strided((1792, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1347 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1348 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1349 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1350 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1351 = rand_strided((1000, 1792), (1792, 1), device='cuda:0', dtype=torch.float32)
    primals_1352 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
