# AOT ID: ['12_forward']
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
# Topologically Sorted Source Nodes: [features_13_conv_1, sigmoid_14, mul_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_13_conv_1 => add_61, mul_89, mul_90, sub_25
#   mul_14 => mul_91
#   sigmoid_14 => sigmoid_13
# Graph fragment:
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_25, %unsqueeze_201), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_203), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %unsqueeze_205), kwargs = {})
#   %add_61 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %unsqueeze_207), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_61,), kwargs = {})
#   %mul_91 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_61, %sigmoid_13), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_13_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_13_conv_4 => add_63, mul_93, mul_94, sub_26
# Graph fragment:
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_94 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_93, %unsqueeze_213), kwargs = {})
#   %add_63 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_94, %unsqueeze_215), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_14_conv_1, sigmoid_15, mul_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_14_conv_1 => add_65, mul_96, mul_97, sub_27
#   mul_15 => mul_98
#   sigmoid_15 => sigmoid_14
# Graph fragment:
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_96 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_97 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_96, %unsqueeze_221), kwargs = {})
#   %add_65 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_97, %unsqueeze_223), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_65,), kwargs = {})
#   %mul_98 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_65, %sigmoid_14), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_14_conv_4, add_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_11 => add_68
#   features_14_conv_4 => add_67, mul_100, mul_101, sub_28
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_100 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_101 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_100, %unsqueeze_229), kwargs = {})
#   %add_67 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_101, %unsqueeze_231), kwargs = {})
#   %add_68 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, %add_67), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_21_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_21_conv_4 => add_102, mul_149, mul_150, sub_42
# Graph fragment:
#   %sub_42 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_42, %unsqueeze_337), kwargs = {})
#   %mul_149 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_42, %unsqueeze_339), kwargs = {})
#   %mul_150 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_149, %unsqueeze_341), kwargs = {})
#   %add_102 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_150, %unsqueeze_343), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_23, mul_23, features_21_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_21_conv_6_avg_pool => mean
#   mul_23 => mul_151
#   sigmoid_23 => sigmoid_22
# Graph fragment:
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_102,), kwargs = {})
#   %mul_151 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_102, %sigmoid_22), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_151, [-1, -2], True), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_24, mul_24], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_24 => mul_152
#   sigmoid_24 => sigmoid_23
# Graph fragment:
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm,), kwargs = {})
#   %mul_152 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, %sigmoid_23), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_23, mul_23, mul_25], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_23 => mul_151
#   mul_25 => mul_153
#   sigmoid_23 => sigmoid_22
# Graph fragment:
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_102,), kwargs = {})
#   %mul_151 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_102, %sigmoid_22), kwargs = {})
#   %mul_153 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_151, %view_1), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_21_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_21_conv_8 => add_104, mul_155, mul_156, sub_43
# Graph fragment:
#   %sub_43 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_345), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_43, %unsqueeze_347), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_155, %unsqueeze_349), kwargs = {})
#   %add_104 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_156, %unsqueeze_351), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_22_conv_1, sigmoid_27, mul_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_22_conv_1 => add_106, mul_158, mul_159, sub_44
#   mul_26 => mul_160
#   sigmoid_27 => sigmoid_25
# Graph fragment:
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %unsqueeze_357), kwargs = {})
#   %add_106 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %unsqueeze_359), kwargs = {})
#   %sigmoid_25 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_106,), kwargs = {})
#   %mul_160 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_106, %sigmoid_25), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_22_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_22_conv_4 => add_108, mul_162, mul_163, sub_45
# Graph fragment:
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_163 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_162, %unsqueeze_365), kwargs = {})
#   %add_108 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_163, %unsqueeze_367), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_29, mul_27, features_22_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_22_conv_6_avg_pool => mean_1
#   mul_27 => mul_164
#   sigmoid_29 => sigmoid_26
# Graph fragment:
#   %sigmoid_26 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_108,), kwargs = {})
#   %mul_164 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_108, %sigmoid_26), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_164, [-1, -2], True), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_31, mul_28], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_28 => mul_165
#   sigmoid_31 => sigmoid_27
# Graph fragment:
#   %sigmoid_27 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_2,), kwargs = {})
#   %mul_165 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_2, %sigmoid_27), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_29, mul_27, mul_29], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_27 => mul_164
#   mul_29 => mul_166
#   sigmoid_29 => sigmoid_26
# Graph fragment:
#   %sigmoid_26 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_108,), kwargs = {})
#   %mul_164 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_108, %sigmoid_26), kwargs = {})
#   %mul_166 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_164, %view_3), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_22_conv_8, add_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_18 => add_111
#   features_22_conv_8 => add_110, mul_168, mul_169, sub_46
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_369), kwargs = {})
#   %mul_168 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_169 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_168, %unsqueeze_373), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_169, %unsqueeze_375), kwargs = {})
#   %add_111 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_104, %add_110), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_37_conv_1, sigmoid_132, mul_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_37_conv_1 => add_211, mul_353, mul_354, sub_89
#   mul_86 => mul_355
#   sigmoid_132 => sigmoid_85
# Graph fragment:
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_89, %unsqueeze_713), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_354 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_353, %unsqueeze_717), kwargs = {})
#   %add_211 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_354, %unsqueeze_719), kwargs = {})
#   %sigmoid_85 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_211,), kwargs = {})
#   %mul_355 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_211, %sigmoid_85), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_37_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_37_conv_4 => add_213, mul_357, mul_358, sub_90
# Graph fragment:
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_721), kwargs = {})
#   %mul_357 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_357, %unsqueeze_725), kwargs = {})
#   %add_213 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_358, %unsqueeze_727), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_134, mul_87, features_37_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_37_conv_6_avg_pool => mean_16
#   mul_87 => mul_359
#   sigmoid_134 => sigmoid_86
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_359 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_86), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_359, [-1, -2], True), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_134, mul_87, mul_89], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_87 => mul_359
#   mul_89 => mul_361
#   sigmoid_134 => sigmoid_86
# Graph fragment:
#   %sigmoid_86 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_213,), kwargs = {})
#   %mul_359 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_213, %sigmoid_86), kwargs = {})
#   %mul_361 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_359, %view_33), kwargs = {})
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


# kernel path: inductor_cache/2w/c2wdvq3zu23t522efzyieh7najdt53ubo53xwgmwqagytnir24xl.py
# Topologically Sorted Source Nodes: [features_37_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_37_conv_8 => add_215, mul_363, mul_364, sub_91
# Graph fragment:
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_729), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_363, %unsqueeze_733), kwargs = {})
#   %add_215 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_364, %unsqueeze_735), kwargs = {})
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


# kernel path: inductor_cache/g6/cg6j5hfpyfx53xszsheyqxdtgvelehdzrtwf6dco5h25ez5p37de.py
# Topologically Sorted Source Nodes: [features_38_conv_1, sigmoid_139, mul_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_38_conv_1 => add_217, mul_366, mul_367, sub_92
#   mul_90 => mul_368
#   sigmoid_139 => sigmoid_89
# Graph fragment:
#   %sub_92 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_92, %unsqueeze_737), kwargs = {})
#   %mul_366 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_92, %unsqueeze_739), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_366, %unsqueeze_741), kwargs = {})
#   %add_217 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_367, %unsqueeze_743), kwargs = {})
#   %sigmoid_89 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_217,), kwargs = {})
#   %mul_368 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_217, %sigmoid_89), kwargs = {})
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
    xnumel = 98304
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


# kernel path: inductor_cache/tb/ctbfljpx4367dvmmkettqxtluowyk62kemsi4werkonddgo6d7uw.py
# Topologically Sorted Source Nodes: [features_38_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_38_conv_4 => add_219, mul_370, mul_371, sub_93
# Graph fragment:
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_745), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_370, %unsqueeze_749), kwargs = {})
#   %add_219 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %unsqueeze_751), kwargs = {})
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
    xnumel = 98304
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


# kernel path: inductor_cache/jb/cjbehzik5vuvi3aefbmfcnlz2ftvdjdcotobejxw2a73h757gkyd.py
# Topologically Sorted Source Nodes: [sigmoid_141, mul_91, features_38_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_38_conv_6_avg_pool => mean_17
#   mul_91 => mul_372
#   sigmoid_141 => sigmoid_90
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_219,), kwargs = {})
#   %mul_372 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_219, %sigmoid_90), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_372, [-1, -2], True), kwargs = {})
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
    xnumel = 6144
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 1536)
    x1 = xindex // 1536
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1536*r2 + 24576*x1), xmask, other=0.0)
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


# kernel path: inductor_cache/yi/cyikymtb6ua6xmj2scz52w4n5hfkq5rh4adipsqqyspwwmrqwo5n.py
# Topologically Sorted Source Nodes: [sigmoid_143, mul_92], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_92 => mul_373
#   sigmoid_143 => sigmoid_91
# Graph fragment:
#   %sigmoid_91 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_34,), kwargs = {})
#   %mul_373 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_34, %sigmoid_91), kwargs = {})
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


# kernel path: inductor_cache/md/cmdqqfwscofztlzn2znuikylpl37ve7o5djrenhaovdkhagm35gi.py
# Topologically Sorted Source Nodes: [sigmoid_141, mul_91, mul_93], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_91 => mul_372
#   mul_93 => mul_374
#   sigmoid_141 => sigmoid_90
# Graph fragment:
#   %sigmoid_90 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_219,), kwargs = {})
#   %mul_372 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_219, %sigmoid_90), kwargs = {})
#   %mul_374 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_372, %view_35), kwargs = {})
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
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x0 = (xindex % 1536)
    x2 = xindex // 24576
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp3 = tl.load(in_ptr0 + (x0 + 1536*x2), None, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tmp4 = tl.sigmoid(tmp3)
    tmp5 = tmp2 * tmp4
    tl.store(in_out_ptr0 + (x3), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/nc/cnclhtcoqt3ywbt4rwubnb3cxcwa4fxt2royjnrz6ijt42635la7.py
# Topologically Sorted Source Nodes: [features_38_conv_8, add_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_33 => add_222
#   features_38_conv_8 => add_221, mul_376, mul_377, sub_94
# Graph fragment:
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %mul_376 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_377 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_376, %unsqueeze_757), kwargs = {})
#   %add_221 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_377, %unsqueeze_759), kwargs = {})
#   %add_222 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_215, %add_221), kwargs = {})
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
    xnumel = 16384
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


# kernel path: inductor_cache/cp/ccpjq3v4mhjnj34z3lvvhezgwbjswt2mpzson5qihigpnbsemc5p.py
# Topologically Sorted Source Nodes: [features_61_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_61_conv_4 => add_380, mul_669, mul_670, sub_162
# Graph fragment:
#   %sub_162 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_162, %unsqueeze_1297), kwargs = {})
#   %mul_669 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_162, %unsqueeze_1299), kwargs = {})
#   %mul_670 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_669, %unsqueeze_1301), kwargs = {})
#   %add_380 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_670, %unsqueeze_1303), kwargs = {})
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


# kernel path: inductor_cache/ze/czeudmnfubl7vwrum2o6bdgcfd6k5cfuqhub2dbo7ngxnig4uwzs.py
# Topologically Sorted Source Nodes: [sigmoid_302, mul_183, features_61_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_61_conv_6_avg_pool => mean_40
#   mul_183 => mul_671
#   sigmoid_302 => sigmoid_182
# Graph fragment:
#   %sigmoid_182 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_380,), kwargs = {})
#   %mul_671 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_380, %sigmoid_182), kwargs = {})
#   %mean_40 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_671, [-1, -2], True), kwargs = {})
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


# kernel path: inductor_cache/c3/cc3xwydq7hp3uvlp2jmxif6egipo34wiblbyr4arod4xwhdrtldd.py
# Topologically Sorted Source Nodes: [sigmoid_302, mul_183, mul_185], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_183 => mul_671
#   mul_185 => mul_673
#   sigmoid_302 => sigmoid_182
# Graph fragment:
#   %sigmoid_182 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_380,), kwargs = {})
#   %mul_671 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_380, %sigmoid_182), kwargs = {})
#   %mul_673 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_671, %view_81), kwargs = {})
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


# kernel path: inductor_cache/em/cem64ca3c3ksh5skxyjt7yolfxl6jyit3wjt7jtodg3b6olw7mda.py
# Topologically Sorted Source Nodes: [features_61_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_61_conv_8 => add_382, mul_675, mul_676, sub_163
# Graph fragment:
#   %sub_163 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_163, %unsqueeze_1305), kwargs = {})
#   %mul_675 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_163, %unsqueeze_1307), kwargs = {})
#   %mul_676 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_675, %unsqueeze_1309), kwargs = {})
#   %add_382 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_676, %unsqueeze_1311), kwargs = {})
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


# kernel path: inductor_cache/ol/colnfvwowfzjhq6nh42kyfg3uzq5df5fcdvekxk4pkujyjwsrwco.py
# Topologically Sorted Source Nodes: [features_62_conv_1, sigmoid_307, mul_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_62_conv_1 => add_384, mul_678, mul_679, sub_164
#   mul_186 => mul_680
#   sigmoid_307 => sigmoid_185
# Graph fragment:
#   %sub_164 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_164, %unsqueeze_1313), kwargs = {})
#   %mul_678 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_164, %unsqueeze_1315), kwargs = {})
#   %mul_679 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_678, %unsqueeze_1317), kwargs = {})
#   %add_384 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_679, %unsqueeze_1319), kwargs = {})
#   %sigmoid_185 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_384,), kwargs = {})
#   %mul_680 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_384, %sigmoid_185), kwargs = {})
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


# kernel path: inductor_cache/rq/crq6xulkflmxnxekkuvduq56rgm4tkxhrd6323qi5ie3on45eiho.py
# Topologically Sorted Source Nodes: [features_62_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_62_conv_4 => add_386, mul_682, mul_683, sub_165
# Graph fragment:
#   %sub_165 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_165, %unsqueeze_1321), kwargs = {})
#   %mul_682 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_165, %unsqueeze_1323), kwargs = {})
#   %mul_683 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_682, %unsqueeze_1325), kwargs = {})
#   %add_386 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_683, %unsqueeze_1327), kwargs = {})
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


# kernel path: inductor_cache/hp/chpgh5zbazieltou2xavu2nwtdgnmxzlrz4hwsin5xt457t5l5ax.py
# Topologically Sorted Source Nodes: [sigmoid_309, mul_187, features_62_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_62_conv_6_avg_pool => mean_41
#   mul_187 => mul_684
#   sigmoid_309 => sigmoid_186
# Graph fragment:
#   %sigmoid_186 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_386,), kwargs = {})
#   %mul_684 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_386, %sigmoid_186), kwargs = {})
#   %mean_41 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_684, [-1, -2], True), kwargs = {})
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


# kernel path: inductor_cache/6g/c6gviwqi7ewq47uw6mhyltabjapvx3le3tvgojmfq2jglvsflwjx.py
# Topologically Sorted Source Nodes: [sigmoid_311, mul_188], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_188 => mul_685
#   sigmoid_311 => sigmoid_187
# Graph fragment:
#   %sigmoid_187 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_82,), kwargs = {})
#   %mul_685 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_82, %sigmoid_187), kwargs = {})
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


# kernel path: inductor_cache/z6/cz65xblr6ad7vdgore4ndiowx3lpp764gqmfctybym347dn26n2t.py
# Topologically Sorted Source Nodes: [sigmoid_309, mul_187, mul_189], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_187 => mul_684
#   mul_189 => mul_686
#   sigmoid_309 => sigmoid_186
# Graph fragment:
#   %sigmoid_186 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_386,), kwargs = {})
#   %mul_684 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_386, %sigmoid_186), kwargs = {})
#   %mul_686 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_684, %view_83), kwargs = {})
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


# kernel path: inductor_cache/hn/chnafoib7aluemiskrrfkztieh3xcrq7vr4lwalr5uqlkq4yhl4x.py
# Topologically Sorted Source Nodes: [features_62_conv_8, add_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_56 => add_389
#   features_62_conv_8 => add_388, mul_688, mul_689, sub_166
# Graph fragment:
#   %sub_166 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_166, %unsqueeze_1329), kwargs = {})
#   %mul_688 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_166, %unsqueeze_1331), kwargs = {})
#   %mul_689 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_688, %unsqueeze_1333), kwargs = {})
#   %add_388 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_689, %unsqueeze_1335), kwargs = {})
#   %add_389 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_382, %add_388), kwargs = {})
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


# kernel path: inductor_cache/l7/cl7skb4cnsajeh355undz5m3jim6mo7k2l7esiogxwcq4od6ciot.py
# Topologically Sorted Source Nodes: [features_93_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_93_conv_8 => add_605, mul_1091, mul_1092, sub_259
# Graph fragment:
#   %sub_259 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_259, %unsqueeze_2073), kwargs = {})
#   %mul_1091 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_259, %unsqueeze_2075), kwargs = {})
#   %mul_1092 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1091, %unsqueeze_2077), kwargs = {})
#   %add_605 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1092, %unsqueeze_2079), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_94_conv_1, sigmoid_531, mul_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   features_94_conv_1 => add_607, mul_1094, mul_1095, sub_260
#   mul_314 => mul_1096
#   sigmoid_531 => sigmoid_313
# Graph fragment:
#   %sub_260 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_260, %unsqueeze_2081), kwargs = {})
#   %mul_1094 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_260, %unsqueeze_2083), kwargs = {})
#   %mul_1095 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1094, %unsqueeze_2085), kwargs = {})
#   %add_607 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1095, %unsqueeze_2087), kwargs = {})
#   %sigmoid_313 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_607,), kwargs = {})
#   %mul_1096 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_607, %sigmoid_313), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_94_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   features_94_conv_4 => add_609, mul_1098, mul_1099, sub_261
# Graph fragment:
#   %sub_261 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_261, %unsqueeze_2089), kwargs = {})
#   %mul_1098 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_261, %unsqueeze_2091), kwargs = {})
#   %mul_1099 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1098, %unsqueeze_2093), kwargs = {})
#   %add_609 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1099, %unsqueeze_2095), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_533, mul_315, features_94_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   features_94_conv_6_avg_pool => mean_73
#   mul_315 => mul_1100
#   sigmoid_533 => sigmoid_314
# Graph fragment:
#   %sigmoid_314 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_609,), kwargs = {})
#   %mul_1100 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_609, %sigmoid_314), kwargs = {})
#   %mean_73 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_1100, [-1, -2], True), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_535, mul_316], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_316 => mul_1101
#   sigmoid_535 => sigmoid_315
# Graph fragment:
#   %sigmoid_315 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%addmm_146,), kwargs = {})
#   %mul_1101 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm_146, %sigmoid_315), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_533, mul_315, mul_317], Original ATen: [aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   mul_315 => mul_1100
#   mul_317 => mul_1102
#   sigmoid_533 => sigmoid_314
# Graph fragment:
#   %sigmoid_314 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_609,), kwargs = {})
#   %mul_1100 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_609, %sigmoid_314), kwargs = {})
#   %mul_1102 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1100, %view_147), kwargs = {})
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
# Topologically Sorted Source Nodes: [features_94_conv_8, add_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   add_87 => add_612
#   features_94_conv_8 => add_611, mul_1104, mul_1105, sub_262
# Graph fragment:
#   %sub_262 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_262, %unsqueeze_2097), kwargs = {})
#   %mul_1104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_262, %unsqueeze_2099), kwargs = {})
#   %mul_1105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1104, %unsqueeze_2101), kwargs = {})
#   %add_611 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1105, %unsqueeze_2103), kwargs = {})
#   %add_612 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_605, %add_611), kwargs = {})
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
#   conv_1 => add_656, mul_1185, mul_1186, sub_281
# Graph fragment:
#   %sub_281 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_281, %unsqueeze_2249), kwargs = {})
#   %mul_1185 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_281, %unsqueeze_2251), kwargs = {})
#   %mul_1186 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1185, %unsqueeze_2253), kwargs = {})
#   %add_656 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1186, %unsqueeze_2255), kwargs = {})
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
# Topologically Sorted Source Nodes: [sigmoid_580, mul_342, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   avgpool => mean_80
#   mul_342 => mul_1187
#   sigmoid_580 => sigmoid_341
# Graph fragment:
#   %sigmoid_341 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%add_656,), kwargs = {})
#   %mul_1187 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_656, %sigmoid_341), kwargs = {})
#   %mean_80 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mul_1187, [-1, -2], True), kwargs = {})
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352, primals_1353, primals_1354, primals_1355, primals_1356, primals_1357, primals_1358, primals_1359, primals_1360, primals_1361, primals_1362, primals_1363, primals_1364, primals_1365, primals_1366, primals_1367, primals_1368, primals_1369, primals_1370, primals_1371, primals_1372, primals_1373, primals_1374, primals_1375, primals_1376, primals_1377, primals_1378, primals_1379, primals_1380, primals_1381, primals_1382, primals_1383, primals_1384, primals_1385, primals_1386, primals_1387, primals_1388, primals_1389, primals_1390, primals_1391, primals_1392, primals_1393, primals_1394, primals_1395, primals_1396, primals_1397, primals_1398, primals_1399, primals_1400, primals_1401, primals_1402, primals_1403, primals_1404, primals_1405, primals_1406, primals_1407, primals_1408, primals_1409, primals_1410, primals_1411, primals_1412, primals_1413, primals_1414, primals_1415, primals_1416, primals_1417, primals_1418, primals_1419, primals_1420, primals_1421, primals_1422, primals_1423, primals_1424, primals_1425, primals_1426, primals_1427, primals_1428, primals_1429, primals_1430, primals_1431, primals_1432, primals_1433, primals_1434, primals_1435, primals_1436, primals_1437, primals_1438, primals_1439, primals_1440, primals_1441, primals_1442, primals_1443, primals_1444, primals_1445, primals_1446, primals_1447, primals_1448, primals_1449, primals_1450, primals_1451, primals_1452, primals_1453, primals_1454, primals_1455, primals_1456, primals_1457, primals_1458, primals_1459, primals_1460, primals_1461, primals_1462, primals_1463, primals_1464, primals_1465, primals_1466, primals_1467, primals_1468, primals_1469, primals_1470, primals_1471, primals_1472, primals_1473, primals_1474, primals_1475, primals_1476, primals_1477, primals_1478, primals_1479, primals_1480, primals_1481, primals_1482, primals_1483, primals_1484, primals_1485, primals_1486, primals_1487, primals_1488, primals_1489, primals_1490, primals_1491, primals_1492, primals_1493, primals_1494, primals_1495, primals_1496, primals_1497, primals_1498, primals_1499, primals_1500, primals_1501, primals_1502, primals_1503, primals_1504, primals_1505, primals_1506, primals_1507, primals_1508, primals_1509, primals_1510, primals_1511, primals_1512, primals_1513, primals_1514, primals_1515, primals_1516, primals_1517, primals_1518, primals_1519, primals_1520, primals_1521, primals_1522, primals_1523, primals_1524, primals_1525, primals_1526, primals_1527, primals_1528, primals_1529, primals_1530, primals_1531, primals_1532, primals_1533, primals_1534, primals_1535, primals_1536, primals_1537, primals_1538, primals_1539, primals_1540, primals_1541, primals_1542, primals_1543, primals_1544, primals_1545, primals_1546, primals_1547, primals_1548, primals_1549, primals_1550, primals_1551, primals_1552, primals_1553, primals_1554, primals_1555, primals_1556, primals_1557, primals_1558, primals_1559, primals_1560, primals_1561, primals_1562, primals_1563, primals_1564, primals_1565, primals_1566, primals_1567, primals_1568, primals_1569, primals_1570, primals_1571, primals_1572, primals_1573, primals_1574, primals_1575, primals_1576, primals_1577, primals_1578, primals_1579, primals_1580, primals_1581, primals_1582, primals_1583, primals_1584, primals_1585, primals_1586, primals_1587, primals_1588, primals_1589, primals_1590, primals_1591, primals_1592, primals_1593, primals_1594, primals_1595, primals_1596, primals_1597, primals_1598, primals_1599, primals_1600, primals_1601, primals_1602, primals_1603, primals_1604, primals_1605, primals_1606, primals_1607, primals_1608, primals_1609, primals_1610, primals_1611, primals_1612, primals_1613, primals_1614, primals_1615, primals_1616, primals_1617, primals_1618, primals_1619, primals_1620, primals_1621, primals_1622, primals_1623, primals_1624, primals_1625, primals_1626, primals_1627, primals_1628, primals_1629, primals_1630, primals_1631, primals_1632, primals_1633, primals_1634, primals_1635, primals_1636, primals_1637, primals_1638, primals_1639, primals_1640, primals_1641, primals_1642, primals_1643, primals_1644, primals_1645, primals_1646, primals_1647, primals_1648, primals_1649, primals_1650, primals_1651, primals_1652, primals_1653, primals_1654, primals_1655, primals_1656, primals_1657, primals_1658, primals_1659, primals_1660, primals_1661, primals_1662, primals_1663, primals_1664, primals_1665, primals_1666, primals_1667, primals_1668, primals_1669, primals_1670, primals_1671, primals_1672, primals_1673, primals_1674, primals_1675, primals_1676, primals_1677, primals_1678, primals_1679, primals_1680, primals_1681, primals_1682, primals_1683, primals_1684, primals_1685, primals_1686, primals_1687, primals_1688, primals_1689, primals_1690, primals_1691, primals_1692, primals_1693, primals_1694, primals_1695, primals_1696, primals_1697, primals_1698, primals_1699, primals_1700, primals_1701, primals_1702, primals_1703, primals_1704, primals_1705, primals_1706, primals_1707, primals_1708, primals_1709, primals_1710, primals_1711, primals_1712, primals_1713, primals_1714, primals_1715, primals_1716, primals_1717, primals_1718, primals_1719, primals_1720, primals_1721, primals_1722, primals_1723, primals_1724, primals_1725, primals_1726, primals_1727, primals_1728, primals_1729, primals_1730, primals_1731, primals_1732, primals_1733 = args
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
    assert_size_stride(primals_122, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (64, ), (1, ))
    assert_size_stride(primals_125, (64, ), (1, ))
    assert_size_stride(primals_126, (64, ), (1, ))
    assert_size_stride(primals_127, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_128, (256, ), (1, ))
    assert_size_stride(primals_129, (256, ), (1, ))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (96, 256, 1, 1), (256, 1, 1, 1))
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
    assert_size_stride(primals_187, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_188, (384, ), (1, ))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (384, ), (1, ))
    assert_size_stride(primals_191, (384, ), (1, ))
    assert_size_stride(primals_192, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_193, (96, ), (1, ))
    assert_size_stride(primals_194, (96, ), (1, ))
    assert_size_stride(primals_195, (96, ), (1, ))
    assert_size_stride(primals_196, (96, ), (1, ))
    assert_size_stride(primals_197, (384, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_198, (384, ), (1, ))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (96, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_203, (96, ), (1, ))
    assert_size_stride(primals_204, (96, ), (1, ))
    assert_size_stride(primals_205, (96, ), (1, ))
    assert_size_stride(primals_206, (96, ), (1, ))
    assert_size_stride(primals_207, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_208, (384, ), (1, ))
    assert_size_stride(primals_209, (384, ), (1, ))
    assert_size_stride(primals_210, (384, ), (1, ))
    assert_size_stride(primals_211, (384, ), (1, ))
    assert_size_stride(primals_212, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_214, (384, ), (1, ))
    assert_size_stride(primals_215, (384, ), (1, ))
    assert_size_stride(primals_216, (384, ), (1, ))
    assert_size_stride(primals_217, (24, 384), (384, 1))
    assert_size_stride(primals_218, (24, ), (1, ))
    assert_size_stride(primals_219, (384, 24), (24, 1))
    assert_size_stride(primals_220, (384, ), (1, ))
    assert_size_stride(primals_221, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_222, (192, ), (1, ))
    assert_size_stride(primals_223, (192, ), (1, ))
    assert_size_stride(primals_224, (192, ), (1, ))
    assert_size_stride(primals_225, (192, ), (1, ))
    assert_size_stride(primals_226, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_227, (768, ), (1, ))
    assert_size_stride(primals_228, (768, ), (1, ))
    assert_size_stride(primals_229, (768, ), (1, ))
    assert_size_stride(primals_230, (768, ), (1, ))
    assert_size_stride(primals_231, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_232, (768, ), (1, ))
    assert_size_stride(primals_233, (768, ), (1, ))
    assert_size_stride(primals_234, (768, ), (1, ))
    assert_size_stride(primals_235, (768, ), (1, ))
    assert_size_stride(primals_236, (48, 768), (768, 1))
    assert_size_stride(primals_237, (48, ), (1, ))
    assert_size_stride(primals_238, (768, 48), (48, 1))
    assert_size_stride(primals_239, (768, ), (1, ))
    assert_size_stride(primals_240, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_241, (192, ), (1, ))
    assert_size_stride(primals_242, (192, ), (1, ))
    assert_size_stride(primals_243, (192, ), (1, ))
    assert_size_stride(primals_244, (192, ), (1, ))
    assert_size_stride(primals_245, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_246, (768, ), (1, ))
    assert_size_stride(primals_247, (768, ), (1, ))
    assert_size_stride(primals_248, (768, ), (1, ))
    assert_size_stride(primals_249, (768, ), (1, ))
    assert_size_stride(primals_250, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_251, (768, ), (1, ))
    assert_size_stride(primals_252, (768, ), (1, ))
    assert_size_stride(primals_253, (768, ), (1, ))
    assert_size_stride(primals_254, (768, ), (1, ))
    assert_size_stride(primals_255, (48, 768), (768, 1))
    assert_size_stride(primals_256, (48, ), (1, ))
    assert_size_stride(primals_257, (768, 48), (48, 1))
    assert_size_stride(primals_258, (768, ), (1, ))
    assert_size_stride(primals_259, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_260, (192, ), (1, ))
    assert_size_stride(primals_261, (192, ), (1, ))
    assert_size_stride(primals_262, (192, ), (1, ))
    assert_size_stride(primals_263, (192, ), (1, ))
    assert_size_stride(primals_264, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_265, (768, ), (1, ))
    assert_size_stride(primals_266, (768, ), (1, ))
    assert_size_stride(primals_267, (768, ), (1, ))
    assert_size_stride(primals_268, (768, ), (1, ))
    assert_size_stride(primals_269, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_270, (768, ), (1, ))
    assert_size_stride(primals_271, (768, ), (1, ))
    assert_size_stride(primals_272, (768, ), (1, ))
    assert_size_stride(primals_273, (768, ), (1, ))
    assert_size_stride(primals_274, (48, 768), (768, 1))
    assert_size_stride(primals_275, (48, ), (1, ))
    assert_size_stride(primals_276, (768, 48), (48, 1))
    assert_size_stride(primals_277, (768, ), (1, ))
    assert_size_stride(primals_278, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_279, (192, ), (1, ))
    assert_size_stride(primals_280, (192, ), (1, ))
    assert_size_stride(primals_281, (192, ), (1, ))
    assert_size_stride(primals_282, (192, ), (1, ))
    assert_size_stride(primals_283, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_284, (768, ), (1, ))
    assert_size_stride(primals_285, (768, ), (1, ))
    assert_size_stride(primals_286, (768, ), (1, ))
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_288, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_289, (768, ), (1, ))
    assert_size_stride(primals_290, (768, ), (1, ))
    assert_size_stride(primals_291, (768, ), (1, ))
    assert_size_stride(primals_292, (768, ), (1, ))
    assert_size_stride(primals_293, (48, 768), (768, 1))
    assert_size_stride(primals_294, (48, ), (1, ))
    assert_size_stride(primals_295, (768, 48), (48, 1))
    assert_size_stride(primals_296, (768, ), (1, ))
    assert_size_stride(primals_297, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_298, (192, ), (1, ))
    assert_size_stride(primals_299, (192, ), (1, ))
    assert_size_stride(primals_300, (192, ), (1, ))
    assert_size_stride(primals_301, (192, ), (1, ))
    assert_size_stride(primals_302, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_303, (768, ), (1, ))
    assert_size_stride(primals_304, (768, ), (1, ))
    assert_size_stride(primals_305, (768, ), (1, ))
    assert_size_stride(primals_306, (768, ), (1, ))
    assert_size_stride(primals_307, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_308, (768, ), (1, ))
    assert_size_stride(primals_309, (768, ), (1, ))
    assert_size_stride(primals_310, (768, ), (1, ))
    assert_size_stride(primals_311, (768, ), (1, ))
    assert_size_stride(primals_312, (48, 768), (768, 1))
    assert_size_stride(primals_313, (48, ), (1, ))
    assert_size_stride(primals_314, (768, 48), (48, 1))
    assert_size_stride(primals_315, (768, ), (1, ))
    assert_size_stride(primals_316, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_317, (192, ), (1, ))
    assert_size_stride(primals_318, (192, ), (1, ))
    assert_size_stride(primals_319, (192, ), (1, ))
    assert_size_stride(primals_320, (192, ), (1, ))
    assert_size_stride(primals_321, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_322, (768, ), (1, ))
    assert_size_stride(primals_323, (768, ), (1, ))
    assert_size_stride(primals_324, (768, ), (1, ))
    assert_size_stride(primals_325, (768, ), (1, ))
    assert_size_stride(primals_326, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_327, (768, ), (1, ))
    assert_size_stride(primals_328, (768, ), (1, ))
    assert_size_stride(primals_329, (768, ), (1, ))
    assert_size_stride(primals_330, (768, ), (1, ))
    assert_size_stride(primals_331, (48, 768), (768, 1))
    assert_size_stride(primals_332, (48, ), (1, ))
    assert_size_stride(primals_333, (768, 48), (48, 1))
    assert_size_stride(primals_334, (768, ), (1, ))
    assert_size_stride(primals_335, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_336, (192, ), (1, ))
    assert_size_stride(primals_337, (192, ), (1, ))
    assert_size_stride(primals_338, (192, ), (1, ))
    assert_size_stride(primals_339, (192, ), (1, ))
    assert_size_stride(primals_340, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_341, (768, ), (1, ))
    assert_size_stride(primals_342, (768, ), (1, ))
    assert_size_stride(primals_343, (768, ), (1, ))
    assert_size_stride(primals_344, (768, ), (1, ))
    assert_size_stride(primals_345, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_346, (768, ), (1, ))
    assert_size_stride(primals_347, (768, ), (1, ))
    assert_size_stride(primals_348, (768, ), (1, ))
    assert_size_stride(primals_349, (768, ), (1, ))
    assert_size_stride(primals_350, (48, 768), (768, 1))
    assert_size_stride(primals_351, (48, ), (1, ))
    assert_size_stride(primals_352, (768, 48), (48, 1))
    assert_size_stride(primals_353, (768, ), (1, ))
    assert_size_stride(primals_354, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_355, (192, ), (1, ))
    assert_size_stride(primals_356, (192, ), (1, ))
    assert_size_stride(primals_357, (192, ), (1, ))
    assert_size_stride(primals_358, (192, ), (1, ))
    assert_size_stride(primals_359, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_360, (768, ), (1, ))
    assert_size_stride(primals_361, (768, ), (1, ))
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (768, ), (1, ))
    assert_size_stride(primals_364, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_365, (768, ), (1, ))
    assert_size_stride(primals_366, (768, ), (1, ))
    assert_size_stride(primals_367, (768, ), (1, ))
    assert_size_stride(primals_368, (768, ), (1, ))
    assert_size_stride(primals_369, (48, 768), (768, 1))
    assert_size_stride(primals_370, (48, ), (1, ))
    assert_size_stride(primals_371, (768, 48), (48, 1))
    assert_size_stride(primals_372, (768, ), (1, ))
    assert_size_stride(primals_373, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_374, (192, ), (1, ))
    assert_size_stride(primals_375, (192, ), (1, ))
    assert_size_stride(primals_376, (192, ), (1, ))
    assert_size_stride(primals_377, (192, ), (1, ))
    assert_size_stride(primals_378, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_379, (768, ), (1, ))
    assert_size_stride(primals_380, (768, ), (1, ))
    assert_size_stride(primals_381, (768, ), (1, ))
    assert_size_stride(primals_382, (768, ), (1, ))
    assert_size_stride(primals_383, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_384, (768, ), (1, ))
    assert_size_stride(primals_385, (768, ), (1, ))
    assert_size_stride(primals_386, (768, ), (1, ))
    assert_size_stride(primals_387, (768, ), (1, ))
    assert_size_stride(primals_388, (48, 768), (768, 1))
    assert_size_stride(primals_389, (48, ), (1, ))
    assert_size_stride(primals_390, (768, 48), (48, 1))
    assert_size_stride(primals_391, (768, ), (1, ))
    assert_size_stride(primals_392, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_393, (192, ), (1, ))
    assert_size_stride(primals_394, (192, ), (1, ))
    assert_size_stride(primals_395, (192, ), (1, ))
    assert_size_stride(primals_396, (192, ), (1, ))
    assert_size_stride(primals_397, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_398, (768, ), (1, ))
    assert_size_stride(primals_399, (768, ), (1, ))
    assert_size_stride(primals_400, (768, ), (1, ))
    assert_size_stride(primals_401, (768, ), (1, ))
    assert_size_stride(primals_402, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_403, (768, ), (1, ))
    assert_size_stride(primals_404, (768, ), (1, ))
    assert_size_stride(primals_405, (768, ), (1, ))
    assert_size_stride(primals_406, (768, ), (1, ))
    assert_size_stride(primals_407, (48, 768), (768, 1))
    assert_size_stride(primals_408, (48, ), (1, ))
    assert_size_stride(primals_409, (768, 48), (48, 1))
    assert_size_stride(primals_410, (768, ), (1, ))
    assert_size_stride(primals_411, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_412, (192, ), (1, ))
    assert_size_stride(primals_413, (192, ), (1, ))
    assert_size_stride(primals_414, (192, ), (1, ))
    assert_size_stride(primals_415, (192, ), (1, ))
    assert_size_stride(primals_416, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_417, (768, ), (1, ))
    assert_size_stride(primals_418, (768, ), (1, ))
    assert_size_stride(primals_419, (768, ), (1, ))
    assert_size_stride(primals_420, (768, ), (1, ))
    assert_size_stride(primals_421, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_422, (768, ), (1, ))
    assert_size_stride(primals_423, (768, ), (1, ))
    assert_size_stride(primals_424, (768, ), (1, ))
    assert_size_stride(primals_425, (768, ), (1, ))
    assert_size_stride(primals_426, (48, 768), (768, 1))
    assert_size_stride(primals_427, (48, ), (1, ))
    assert_size_stride(primals_428, (768, 48), (48, 1))
    assert_size_stride(primals_429, (768, ), (1, ))
    assert_size_stride(primals_430, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_431, (192, ), (1, ))
    assert_size_stride(primals_432, (192, ), (1, ))
    assert_size_stride(primals_433, (192, ), (1, ))
    assert_size_stride(primals_434, (192, ), (1, ))
    assert_size_stride(primals_435, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_436, (768, ), (1, ))
    assert_size_stride(primals_437, (768, ), (1, ))
    assert_size_stride(primals_438, (768, ), (1, ))
    assert_size_stride(primals_439, (768, ), (1, ))
    assert_size_stride(primals_440, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_441, (768, ), (1, ))
    assert_size_stride(primals_442, (768, ), (1, ))
    assert_size_stride(primals_443, (768, ), (1, ))
    assert_size_stride(primals_444, (768, ), (1, ))
    assert_size_stride(primals_445, (48, 768), (768, 1))
    assert_size_stride(primals_446, (48, ), (1, ))
    assert_size_stride(primals_447, (768, 48), (48, 1))
    assert_size_stride(primals_448, (768, ), (1, ))
    assert_size_stride(primals_449, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_450, (192, ), (1, ))
    assert_size_stride(primals_451, (192, ), (1, ))
    assert_size_stride(primals_452, (192, ), (1, ))
    assert_size_stride(primals_453, (192, ), (1, ))
    assert_size_stride(primals_454, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_455, (768, ), (1, ))
    assert_size_stride(primals_456, (768, ), (1, ))
    assert_size_stride(primals_457, (768, ), (1, ))
    assert_size_stride(primals_458, (768, ), (1, ))
    assert_size_stride(primals_459, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_460, (768, ), (1, ))
    assert_size_stride(primals_461, (768, ), (1, ))
    assert_size_stride(primals_462, (768, ), (1, ))
    assert_size_stride(primals_463, (768, ), (1, ))
    assert_size_stride(primals_464, (48, 768), (768, 1))
    assert_size_stride(primals_465, (48, ), (1, ))
    assert_size_stride(primals_466, (768, 48), (48, 1))
    assert_size_stride(primals_467, (768, ), (1, ))
    assert_size_stride(primals_468, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_469, (192, ), (1, ))
    assert_size_stride(primals_470, (192, ), (1, ))
    assert_size_stride(primals_471, (192, ), (1, ))
    assert_size_stride(primals_472, (192, ), (1, ))
    assert_size_stride(primals_473, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_474, (768, ), (1, ))
    assert_size_stride(primals_475, (768, ), (1, ))
    assert_size_stride(primals_476, (768, ), (1, ))
    assert_size_stride(primals_477, (768, ), (1, ))
    assert_size_stride(primals_478, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_479, (768, ), (1, ))
    assert_size_stride(primals_480, (768, ), (1, ))
    assert_size_stride(primals_481, (768, ), (1, ))
    assert_size_stride(primals_482, (768, ), (1, ))
    assert_size_stride(primals_483, (48, 768), (768, 1))
    assert_size_stride(primals_484, (48, ), (1, ))
    assert_size_stride(primals_485, (768, 48), (48, 1))
    assert_size_stride(primals_486, (768, ), (1, ))
    assert_size_stride(primals_487, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_488, (192, ), (1, ))
    assert_size_stride(primals_489, (192, ), (1, ))
    assert_size_stride(primals_490, (192, ), (1, ))
    assert_size_stride(primals_491, (192, ), (1, ))
    assert_size_stride(primals_492, (768, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_493, (768, ), (1, ))
    assert_size_stride(primals_494, (768, ), (1, ))
    assert_size_stride(primals_495, (768, ), (1, ))
    assert_size_stride(primals_496, (768, ), (1, ))
    assert_size_stride(primals_497, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_498, (768, ), (1, ))
    assert_size_stride(primals_499, (768, ), (1, ))
    assert_size_stride(primals_500, (768, ), (1, ))
    assert_size_stride(primals_501, (768, ), (1, ))
    assert_size_stride(primals_502, (48, 768), (768, 1))
    assert_size_stride(primals_503, (48, ), (1, ))
    assert_size_stride(primals_504, (768, 48), (48, 1))
    assert_size_stride(primals_505, (768, ), (1, ))
    assert_size_stride(primals_506, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_507, (192, ), (1, ))
    assert_size_stride(primals_508, (192, ), (1, ))
    assert_size_stride(primals_509, (192, ), (1, ))
    assert_size_stride(primals_510, (192, ), (1, ))
    assert_size_stride(primals_511, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_512, (1152, ), (1, ))
    assert_size_stride(primals_513, (1152, ), (1, ))
    assert_size_stride(primals_514, (1152, ), (1, ))
    assert_size_stride(primals_515, (1152, ), (1, ))
    assert_size_stride(primals_516, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_517, (1152, ), (1, ))
    assert_size_stride(primals_518, (1152, ), (1, ))
    assert_size_stride(primals_519, (1152, ), (1, ))
    assert_size_stride(primals_520, (1152, ), (1, ))
    assert_size_stride(primals_521, (48, 1152), (1152, 1))
    assert_size_stride(primals_522, (48, ), (1, ))
    assert_size_stride(primals_523, (1152, 48), (48, 1))
    assert_size_stride(primals_524, (1152, ), (1, ))
    assert_size_stride(primals_525, (256, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_526, (256, ), (1, ))
    assert_size_stride(primals_527, (256, ), (1, ))
    assert_size_stride(primals_528, (256, ), (1, ))
    assert_size_stride(primals_529, (256, ), (1, ))
    assert_size_stride(primals_530, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_531, (1536, ), (1, ))
    assert_size_stride(primals_532, (1536, ), (1, ))
    assert_size_stride(primals_533, (1536, ), (1, ))
    assert_size_stride(primals_534, (1536, ), (1, ))
    assert_size_stride(primals_535, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_536, (1536, ), (1, ))
    assert_size_stride(primals_537, (1536, ), (1, ))
    assert_size_stride(primals_538, (1536, ), (1, ))
    assert_size_stride(primals_539, (1536, ), (1, ))
    assert_size_stride(primals_540, (64, 1536), (1536, 1))
    assert_size_stride(primals_541, (64, ), (1, ))
    assert_size_stride(primals_542, (1536, 64), (64, 1))
    assert_size_stride(primals_543, (1536, ), (1, ))
    assert_size_stride(primals_544, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_545, (256, ), (1, ))
    assert_size_stride(primals_546, (256, ), (1, ))
    assert_size_stride(primals_547, (256, ), (1, ))
    assert_size_stride(primals_548, (256, ), (1, ))
    assert_size_stride(primals_549, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_550, (1536, ), (1, ))
    assert_size_stride(primals_551, (1536, ), (1, ))
    assert_size_stride(primals_552, (1536, ), (1, ))
    assert_size_stride(primals_553, (1536, ), (1, ))
    assert_size_stride(primals_554, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_555, (1536, ), (1, ))
    assert_size_stride(primals_556, (1536, ), (1, ))
    assert_size_stride(primals_557, (1536, ), (1, ))
    assert_size_stride(primals_558, (1536, ), (1, ))
    assert_size_stride(primals_559, (64, 1536), (1536, 1))
    assert_size_stride(primals_560, (64, ), (1, ))
    assert_size_stride(primals_561, (1536, 64), (64, 1))
    assert_size_stride(primals_562, (1536, ), (1, ))
    assert_size_stride(primals_563, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_564, (256, ), (1, ))
    assert_size_stride(primals_565, (256, ), (1, ))
    assert_size_stride(primals_566, (256, ), (1, ))
    assert_size_stride(primals_567, (256, ), (1, ))
    assert_size_stride(primals_568, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_569, (1536, ), (1, ))
    assert_size_stride(primals_570, (1536, ), (1, ))
    assert_size_stride(primals_571, (1536, ), (1, ))
    assert_size_stride(primals_572, (1536, ), (1, ))
    assert_size_stride(primals_573, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_574, (1536, ), (1, ))
    assert_size_stride(primals_575, (1536, ), (1, ))
    assert_size_stride(primals_576, (1536, ), (1, ))
    assert_size_stride(primals_577, (1536, ), (1, ))
    assert_size_stride(primals_578, (64, 1536), (1536, 1))
    assert_size_stride(primals_579, (64, ), (1, ))
    assert_size_stride(primals_580, (1536, 64), (64, 1))
    assert_size_stride(primals_581, (1536, ), (1, ))
    assert_size_stride(primals_582, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_583, (256, ), (1, ))
    assert_size_stride(primals_584, (256, ), (1, ))
    assert_size_stride(primals_585, (256, ), (1, ))
    assert_size_stride(primals_586, (256, ), (1, ))
    assert_size_stride(primals_587, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_588, (1536, ), (1, ))
    assert_size_stride(primals_589, (1536, ), (1, ))
    assert_size_stride(primals_590, (1536, ), (1, ))
    assert_size_stride(primals_591, (1536, ), (1, ))
    assert_size_stride(primals_592, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_593, (1536, ), (1, ))
    assert_size_stride(primals_594, (1536, ), (1, ))
    assert_size_stride(primals_595, (1536, ), (1, ))
    assert_size_stride(primals_596, (1536, ), (1, ))
    assert_size_stride(primals_597, (64, 1536), (1536, 1))
    assert_size_stride(primals_598, (64, ), (1, ))
    assert_size_stride(primals_599, (1536, 64), (64, 1))
    assert_size_stride(primals_600, (1536, ), (1, ))
    assert_size_stride(primals_601, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_602, (256, ), (1, ))
    assert_size_stride(primals_603, (256, ), (1, ))
    assert_size_stride(primals_604, (256, ), (1, ))
    assert_size_stride(primals_605, (256, ), (1, ))
    assert_size_stride(primals_606, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_607, (1536, ), (1, ))
    assert_size_stride(primals_608, (1536, ), (1, ))
    assert_size_stride(primals_609, (1536, ), (1, ))
    assert_size_stride(primals_610, (1536, ), (1, ))
    assert_size_stride(primals_611, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_612, (1536, ), (1, ))
    assert_size_stride(primals_613, (1536, ), (1, ))
    assert_size_stride(primals_614, (1536, ), (1, ))
    assert_size_stride(primals_615, (1536, ), (1, ))
    assert_size_stride(primals_616, (64, 1536), (1536, 1))
    assert_size_stride(primals_617, (64, ), (1, ))
    assert_size_stride(primals_618, (1536, 64), (64, 1))
    assert_size_stride(primals_619, (1536, ), (1, ))
    assert_size_stride(primals_620, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_621, (256, ), (1, ))
    assert_size_stride(primals_622, (256, ), (1, ))
    assert_size_stride(primals_623, (256, ), (1, ))
    assert_size_stride(primals_624, (256, ), (1, ))
    assert_size_stride(primals_625, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_626, (1536, ), (1, ))
    assert_size_stride(primals_627, (1536, ), (1, ))
    assert_size_stride(primals_628, (1536, ), (1, ))
    assert_size_stride(primals_629, (1536, ), (1, ))
    assert_size_stride(primals_630, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_631, (1536, ), (1, ))
    assert_size_stride(primals_632, (1536, ), (1, ))
    assert_size_stride(primals_633, (1536, ), (1, ))
    assert_size_stride(primals_634, (1536, ), (1, ))
    assert_size_stride(primals_635, (64, 1536), (1536, 1))
    assert_size_stride(primals_636, (64, ), (1, ))
    assert_size_stride(primals_637, (1536, 64), (64, 1))
    assert_size_stride(primals_638, (1536, ), (1, ))
    assert_size_stride(primals_639, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_640, (256, ), (1, ))
    assert_size_stride(primals_641, (256, ), (1, ))
    assert_size_stride(primals_642, (256, ), (1, ))
    assert_size_stride(primals_643, (256, ), (1, ))
    assert_size_stride(primals_644, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_645, (1536, ), (1, ))
    assert_size_stride(primals_646, (1536, ), (1, ))
    assert_size_stride(primals_647, (1536, ), (1, ))
    assert_size_stride(primals_648, (1536, ), (1, ))
    assert_size_stride(primals_649, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_650, (1536, ), (1, ))
    assert_size_stride(primals_651, (1536, ), (1, ))
    assert_size_stride(primals_652, (1536, ), (1, ))
    assert_size_stride(primals_653, (1536, ), (1, ))
    assert_size_stride(primals_654, (64, 1536), (1536, 1))
    assert_size_stride(primals_655, (64, ), (1, ))
    assert_size_stride(primals_656, (1536, 64), (64, 1))
    assert_size_stride(primals_657, (1536, ), (1, ))
    assert_size_stride(primals_658, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_659, (256, ), (1, ))
    assert_size_stride(primals_660, (256, ), (1, ))
    assert_size_stride(primals_661, (256, ), (1, ))
    assert_size_stride(primals_662, (256, ), (1, ))
    assert_size_stride(primals_663, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_664, (1536, ), (1, ))
    assert_size_stride(primals_665, (1536, ), (1, ))
    assert_size_stride(primals_666, (1536, ), (1, ))
    assert_size_stride(primals_667, (1536, ), (1, ))
    assert_size_stride(primals_668, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_669, (1536, ), (1, ))
    assert_size_stride(primals_670, (1536, ), (1, ))
    assert_size_stride(primals_671, (1536, ), (1, ))
    assert_size_stride(primals_672, (1536, ), (1, ))
    assert_size_stride(primals_673, (64, 1536), (1536, 1))
    assert_size_stride(primals_674, (64, ), (1, ))
    assert_size_stride(primals_675, (1536, 64), (64, 1))
    assert_size_stride(primals_676, (1536, ), (1, ))
    assert_size_stride(primals_677, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_678, (256, ), (1, ))
    assert_size_stride(primals_679, (256, ), (1, ))
    assert_size_stride(primals_680, (256, ), (1, ))
    assert_size_stride(primals_681, (256, ), (1, ))
    assert_size_stride(primals_682, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_683, (1536, ), (1, ))
    assert_size_stride(primals_684, (1536, ), (1, ))
    assert_size_stride(primals_685, (1536, ), (1, ))
    assert_size_stride(primals_686, (1536, ), (1, ))
    assert_size_stride(primals_687, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_688, (1536, ), (1, ))
    assert_size_stride(primals_689, (1536, ), (1, ))
    assert_size_stride(primals_690, (1536, ), (1, ))
    assert_size_stride(primals_691, (1536, ), (1, ))
    assert_size_stride(primals_692, (64, 1536), (1536, 1))
    assert_size_stride(primals_693, (64, ), (1, ))
    assert_size_stride(primals_694, (1536, 64), (64, 1))
    assert_size_stride(primals_695, (1536, ), (1, ))
    assert_size_stride(primals_696, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_697, (256, ), (1, ))
    assert_size_stride(primals_698, (256, ), (1, ))
    assert_size_stride(primals_699, (256, ), (1, ))
    assert_size_stride(primals_700, (256, ), (1, ))
    assert_size_stride(primals_701, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_702, (1536, ), (1, ))
    assert_size_stride(primals_703, (1536, ), (1, ))
    assert_size_stride(primals_704, (1536, ), (1, ))
    assert_size_stride(primals_705, (1536, ), (1, ))
    assert_size_stride(primals_706, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_707, (1536, ), (1, ))
    assert_size_stride(primals_708, (1536, ), (1, ))
    assert_size_stride(primals_709, (1536, ), (1, ))
    assert_size_stride(primals_710, (1536, ), (1, ))
    assert_size_stride(primals_711, (64, 1536), (1536, 1))
    assert_size_stride(primals_712, (64, ), (1, ))
    assert_size_stride(primals_713, (1536, 64), (64, 1))
    assert_size_stride(primals_714, (1536, ), (1, ))
    assert_size_stride(primals_715, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_716, (256, ), (1, ))
    assert_size_stride(primals_717, (256, ), (1, ))
    assert_size_stride(primals_718, (256, ), (1, ))
    assert_size_stride(primals_719, (256, ), (1, ))
    assert_size_stride(primals_720, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_721, (1536, ), (1, ))
    assert_size_stride(primals_722, (1536, ), (1, ))
    assert_size_stride(primals_723, (1536, ), (1, ))
    assert_size_stride(primals_724, (1536, ), (1, ))
    assert_size_stride(primals_725, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_726, (1536, ), (1, ))
    assert_size_stride(primals_727, (1536, ), (1, ))
    assert_size_stride(primals_728, (1536, ), (1, ))
    assert_size_stride(primals_729, (1536, ), (1, ))
    assert_size_stride(primals_730, (64, 1536), (1536, 1))
    assert_size_stride(primals_731, (64, ), (1, ))
    assert_size_stride(primals_732, (1536, 64), (64, 1))
    assert_size_stride(primals_733, (1536, ), (1, ))
    assert_size_stride(primals_734, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_735, (256, ), (1, ))
    assert_size_stride(primals_736, (256, ), (1, ))
    assert_size_stride(primals_737, (256, ), (1, ))
    assert_size_stride(primals_738, (256, ), (1, ))
    assert_size_stride(primals_739, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_740, (1536, ), (1, ))
    assert_size_stride(primals_741, (1536, ), (1, ))
    assert_size_stride(primals_742, (1536, ), (1, ))
    assert_size_stride(primals_743, (1536, ), (1, ))
    assert_size_stride(primals_744, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_745, (1536, ), (1, ))
    assert_size_stride(primals_746, (1536, ), (1, ))
    assert_size_stride(primals_747, (1536, ), (1, ))
    assert_size_stride(primals_748, (1536, ), (1, ))
    assert_size_stride(primals_749, (64, 1536), (1536, 1))
    assert_size_stride(primals_750, (64, ), (1, ))
    assert_size_stride(primals_751, (1536, 64), (64, 1))
    assert_size_stride(primals_752, (1536, ), (1, ))
    assert_size_stride(primals_753, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_754, (256, ), (1, ))
    assert_size_stride(primals_755, (256, ), (1, ))
    assert_size_stride(primals_756, (256, ), (1, ))
    assert_size_stride(primals_757, (256, ), (1, ))
    assert_size_stride(primals_758, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_759, (1536, ), (1, ))
    assert_size_stride(primals_760, (1536, ), (1, ))
    assert_size_stride(primals_761, (1536, ), (1, ))
    assert_size_stride(primals_762, (1536, ), (1, ))
    assert_size_stride(primals_763, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_764, (1536, ), (1, ))
    assert_size_stride(primals_765, (1536, ), (1, ))
    assert_size_stride(primals_766, (1536, ), (1, ))
    assert_size_stride(primals_767, (1536, ), (1, ))
    assert_size_stride(primals_768, (64, 1536), (1536, 1))
    assert_size_stride(primals_769, (64, ), (1, ))
    assert_size_stride(primals_770, (1536, 64), (64, 1))
    assert_size_stride(primals_771, (1536, ), (1, ))
    assert_size_stride(primals_772, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_773, (256, ), (1, ))
    assert_size_stride(primals_774, (256, ), (1, ))
    assert_size_stride(primals_775, (256, ), (1, ))
    assert_size_stride(primals_776, (256, ), (1, ))
    assert_size_stride(primals_777, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_778, (1536, ), (1, ))
    assert_size_stride(primals_779, (1536, ), (1, ))
    assert_size_stride(primals_780, (1536, ), (1, ))
    assert_size_stride(primals_781, (1536, ), (1, ))
    assert_size_stride(primals_782, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_783, (1536, ), (1, ))
    assert_size_stride(primals_784, (1536, ), (1, ))
    assert_size_stride(primals_785, (1536, ), (1, ))
    assert_size_stride(primals_786, (1536, ), (1, ))
    assert_size_stride(primals_787, (64, 1536), (1536, 1))
    assert_size_stride(primals_788, (64, ), (1, ))
    assert_size_stride(primals_789, (1536, 64), (64, 1))
    assert_size_stride(primals_790, (1536, ), (1, ))
    assert_size_stride(primals_791, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_792, (256, ), (1, ))
    assert_size_stride(primals_793, (256, ), (1, ))
    assert_size_stride(primals_794, (256, ), (1, ))
    assert_size_stride(primals_795, (256, ), (1, ))
    assert_size_stride(primals_796, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_797, (1536, ), (1, ))
    assert_size_stride(primals_798, (1536, ), (1, ))
    assert_size_stride(primals_799, (1536, ), (1, ))
    assert_size_stride(primals_800, (1536, ), (1, ))
    assert_size_stride(primals_801, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_802, (1536, ), (1, ))
    assert_size_stride(primals_803, (1536, ), (1, ))
    assert_size_stride(primals_804, (1536, ), (1, ))
    assert_size_stride(primals_805, (1536, ), (1, ))
    assert_size_stride(primals_806, (64, 1536), (1536, 1))
    assert_size_stride(primals_807, (64, ), (1, ))
    assert_size_stride(primals_808, (1536, 64), (64, 1))
    assert_size_stride(primals_809, (1536, ), (1, ))
    assert_size_stride(primals_810, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_811, (256, ), (1, ))
    assert_size_stride(primals_812, (256, ), (1, ))
    assert_size_stride(primals_813, (256, ), (1, ))
    assert_size_stride(primals_814, (256, ), (1, ))
    assert_size_stride(primals_815, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_816, (1536, ), (1, ))
    assert_size_stride(primals_817, (1536, ), (1, ))
    assert_size_stride(primals_818, (1536, ), (1, ))
    assert_size_stride(primals_819, (1536, ), (1, ))
    assert_size_stride(primals_820, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_821, (1536, ), (1, ))
    assert_size_stride(primals_822, (1536, ), (1, ))
    assert_size_stride(primals_823, (1536, ), (1, ))
    assert_size_stride(primals_824, (1536, ), (1, ))
    assert_size_stride(primals_825, (64, 1536), (1536, 1))
    assert_size_stride(primals_826, (64, ), (1, ))
    assert_size_stride(primals_827, (1536, 64), (64, 1))
    assert_size_stride(primals_828, (1536, ), (1, ))
    assert_size_stride(primals_829, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_830, (256, ), (1, ))
    assert_size_stride(primals_831, (256, ), (1, ))
    assert_size_stride(primals_832, (256, ), (1, ))
    assert_size_stride(primals_833, (256, ), (1, ))
    assert_size_stride(primals_834, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_835, (1536, ), (1, ))
    assert_size_stride(primals_836, (1536, ), (1, ))
    assert_size_stride(primals_837, (1536, ), (1, ))
    assert_size_stride(primals_838, (1536, ), (1, ))
    assert_size_stride(primals_839, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_840, (1536, ), (1, ))
    assert_size_stride(primals_841, (1536, ), (1, ))
    assert_size_stride(primals_842, (1536, ), (1, ))
    assert_size_stride(primals_843, (1536, ), (1, ))
    assert_size_stride(primals_844, (64, 1536), (1536, 1))
    assert_size_stride(primals_845, (64, ), (1, ))
    assert_size_stride(primals_846, (1536, 64), (64, 1))
    assert_size_stride(primals_847, (1536, ), (1, ))
    assert_size_stride(primals_848, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_849, (256, ), (1, ))
    assert_size_stride(primals_850, (256, ), (1, ))
    assert_size_stride(primals_851, (256, ), (1, ))
    assert_size_stride(primals_852, (256, ), (1, ))
    assert_size_stride(primals_853, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_854, (1536, ), (1, ))
    assert_size_stride(primals_855, (1536, ), (1, ))
    assert_size_stride(primals_856, (1536, ), (1, ))
    assert_size_stride(primals_857, (1536, ), (1, ))
    assert_size_stride(primals_858, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_859, (1536, ), (1, ))
    assert_size_stride(primals_860, (1536, ), (1, ))
    assert_size_stride(primals_861, (1536, ), (1, ))
    assert_size_stride(primals_862, (1536, ), (1, ))
    assert_size_stride(primals_863, (64, 1536), (1536, 1))
    assert_size_stride(primals_864, (64, ), (1, ))
    assert_size_stride(primals_865, (1536, 64), (64, 1))
    assert_size_stride(primals_866, (1536, ), (1, ))
    assert_size_stride(primals_867, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_868, (256, ), (1, ))
    assert_size_stride(primals_869, (256, ), (1, ))
    assert_size_stride(primals_870, (256, ), (1, ))
    assert_size_stride(primals_871, (256, ), (1, ))
    assert_size_stride(primals_872, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_873, (1536, ), (1, ))
    assert_size_stride(primals_874, (1536, ), (1, ))
    assert_size_stride(primals_875, (1536, ), (1, ))
    assert_size_stride(primals_876, (1536, ), (1, ))
    assert_size_stride(primals_877, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_878, (1536, ), (1, ))
    assert_size_stride(primals_879, (1536, ), (1, ))
    assert_size_stride(primals_880, (1536, ), (1, ))
    assert_size_stride(primals_881, (1536, ), (1, ))
    assert_size_stride(primals_882, (64, 1536), (1536, 1))
    assert_size_stride(primals_883, (64, ), (1, ))
    assert_size_stride(primals_884, (1536, 64), (64, 1))
    assert_size_stride(primals_885, (1536, ), (1, ))
    assert_size_stride(primals_886, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_887, (256, ), (1, ))
    assert_size_stride(primals_888, (256, ), (1, ))
    assert_size_stride(primals_889, (256, ), (1, ))
    assert_size_stride(primals_890, (256, ), (1, ))
    assert_size_stride(primals_891, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_892, (1536, ), (1, ))
    assert_size_stride(primals_893, (1536, ), (1, ))
    assert_size_stride(primals_894, (1536, ), (1, ))
    assert_size_stride(primals_895, (1536, ), (1, ))
    assert_size_stride(primals_896, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_897, (1536, ), (1, ))
    assert_size_stride(primals_898, (1536, ), (1, ))
    assert_size_stride(primals_899, (1536, ), (1, ))
    assert_size_stride(primals_900, (1536, ), (1, ))
    assert_size_stride(primals_901, (64, 1536), (1536, 1))
    assert_size_stride(primals_902, (64, ), (1, ))
    assert_size_stride(primals_903, (1536, 64), (64, 1))
    assert_size_stride(primals_904, (1536, ), (1, ))
    assert_size_stride(primals_905, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_906, (256, ), (1, ))
    assert_size_stride(primals_907, (256, ), (1, ))
    assert_size_stride(primals_908, (256, ), (1, ))
    assert_size_stride(primals_909, (256, ), (1, ))
    assert_size_stride(primals_910, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_911, (1536, ), (1, ))
    assert_size_stride(primals_912, (1536, ), (1, ))
    assert_size_stride(primals_913, (1536, ), (1, ))
    assert_size_stride(primals_914, (1536, ), (1, ))
    assert_size_stride(primals_915, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_916, (1536, ), (1, ))
    assert_size_stride(primals_917, (1536, ), (1, ))
    assert_size_stride(primals_918, (1536, ), (1, ))
    assert_size_stride(primals_919, (1536, ), (1, ))
    assert_size_stride(primals_920, (64, 1536), (1536, 1))
    assert_size_stride(primals_921, (64, ), (1, ))
    assert_size_stride(primals_922, (1536, 64), (64, 1))
    assert_size_stride(primals_923, (1536, ), (1, ))
    assert_size_stride(primals_924, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_925, (256, ), (1, ))
    assert_size_stride(primals_926, (256, ), (1, ))
    assert_size_stride(primals_927, (256, ), (1, ))
    assert_size_stride(primals_928, (256, ), (1, ))
    assert_size_stride(primals_929, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_930, (1536, ), (1, ))
    assert_size_stride(primals_931, (1536, ), (1, ))
    assert_size_stride(primals_932, (1536, ), (1, ))
    assert_size_stride(primals_933, (1536, ), (1, ))
    assert_size_stride(primals_934, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_935, (1536, ), (1, ))
    assert_size_stride(primals_936, (1536, ), (1, ))
    assert_size_stride(primals_937, (1536, ), (1, ))
    assert_size_stride(primals_938, (1536, ), (1, ))
    assert_size_stride(primals_939, (64, 1536), (1536, 1))
    assert_size_stride(primals_940, (64, ), (1, ))
    assert_size_stride(primals_941, (1536, 64), (64, 1))
    assert_size_stride(primals_942, (1536, ), (1, ))
    assert_size_stride(primals_943, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_944, (256, ), (1, ))
    assert_size_stride(primals_945, (256, ), (1, ))
    assert_size_stride(primals_946, (256, ), (1, ))
    assert_size_stride(primals_947, (256, ), (1, ))
    assert_size_stride(primals_948, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_949, (1536, ), (1, ))
    assert_size_stride(primals_950, (1536, ), (1, ))
    assert_size_stride(primals_951, (1536, ), (1, ))
    assert_size_stride(primals_952, (1536, ), (1, ))
    assert_size_stride(primals_953, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_954, (1536, ), (1, ))
    assert_size_stride(primals_955, (1536, ), (1, ))
    assert_size_stride(primals_956, (1536, ), (1, ))
    assert_size_stride(primals_957, (1536, ), (1, ))
    assert_size_stride(primals_958, (64, 1536), (1536, 1))
    assert_size_stride(primals_959, (64, ), (1, ))
    assert_size_stride(primals_960, (1536, 64), (64, 1))
    assert_size_stride(primals_961, (1536, ), (1, ))
    assert_size_stride(primals_962, (256, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_963, (256, ), (1, ))
    assert_size_stride(primals_964, (256, ), (1, ))
    assert_size_stride(primals_965, (256, ), (1, ))
    assert_size_stride(primals_966, (256, ), (1, ))
    assert_size_stride(primals_967, (1536, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_968, (1536, ), (1, ))
    assert_size_stride(primals_969, (1536, ), (1, ))
    assert_size_stride(primals_970, (1536, ), (1, ))
    assert_size_stride(primals_971, (1536, ), (1, ))
    assert_size_stride(primals_972, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_973, (1536, ), (1, ))
    assert_size_stride(primals_974, (1536, ), (1, ))
    assert_size_stride(primals_975, (1536, ), (1, ))
    assert_size_stride(primals_976, (1536, ), (1, ))
    assert_size_stride(primals_977, (64, 1536), (1536, 1))
    assert_size_stride(primals_978, (64, ), (1, ))
    assert_size_stride(primals_979, (1536, 64), (64, 1))
    assert_size_stride(primals_980, (1536, ), (1, ))
    assert_size_stride(primals_981, (512, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_982, (512, ), (1, ))
    assert_size_stride(primals_983, (512, ), (1, ))
    assert_size_stride(primals_984, (512, ), (1, ))
    assert_size_stride(primals_985, (512, ), (1, ))
    assert_size_stride(primals_986, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_987, (3072, ), (1, ))
    assert_size_stride(primals_988, (3072, ), (1, ))
    assert_size_stride(primals_989, (3072, ), (1, ))
    assert_size_stride(primals_990, (3072, ), (1, ))
    assert_size_stride(primals_991, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_992, (3072, ), (1, ))
    assert_size_stride(primals_993, (3072, ), (1, ))
    assert_size_stride(primals_994, (3072, ), (1, ))
    assert_size_stride(primals_995, (3072, ), (1, ))
    assert_size_stride(primals_996, (128, 3072), (3072, 1))
    assert_size_stride(primals_997, (128, ), (1, ))
    assert_size_stride(primals_998, (3072, 128), (128, 1))
    assert_size_stride(primals_999, (3072, ), (1, ))
    assert_size_stride(primals_1000, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1001, (512, ), (1, ))
    assert_size_stride(primals_1002, (512, ), (1, ))
    assert_size_stride(primals_1003, (512, ), (1, ))
    assert_size_stride(primals_1004, (512, ), (1, ))
    assert_size_stride(primals_1005, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1006, (3072, ), (1, ))
    assert_size_stride(primals_1007, (3072, ), (1, ))
    assert_size_stride(primals_1008, (3072, ), (1, ))
    assert_size_stride(primals_1009, (3072, ), (1, ))
    assert_size_stride(primals_1010, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1011, (3072, ), (1, ))
    assert_size_stride(primals_1012, (3072, ), (1, ))
    assert_size_stride(primals_1013, (3072, ), (1, ))
    assert_size_stride(primals_1014, (3072, ), (1, ))
    assert_size_stride(primals_1015, (128, 3072), (3072, 1))
    assert_size_stride(primals_1016, (128, ), (1, ))
    assert_size_stride(primals_1017, (3072, 128), (128, 1))
    assert_size_stride(primals_1018, (3072, ), (1, ))
    assert_size_stride(primals_1019, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1020, (512, ), (1, ))
    assert_size_stride(primals_1021, (512, ), (1, ))
    assert_size_stride(primals_1022, (512, ), (1, ))
    assert_size_stride(primals_1023, (512, ), (1, ))
    assert_size_stride(primals_1024, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1025, (3072, ), (1, ))
    assert_size_stride(primals_1026, (3072, ), (1, ))
    assert_size_stride(primals_1027, (3072, ), (1, ))
    assert_size_stride(primals_1028, (3072, ), (1, ))
    assert_size_stride(primals_1029, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1030, (3072, ), (1, ))
    assert_size_stride(primals_1031, (3072, ), (1, ))
    assert_size_stride(primals_1032, (3072, ), (1, ))
    assert_size_stride(primals_1033, (3072, ), (1, ))
    assert_size_stride(primals_1034, (128, 3072), (3072, 1))
    assert_size_stride(primals_1035, (128, ), (1, ))
    assert_size_stride(primals_1036, (3072, 128), (128, 1))
    assert_size_stride(primals_1037, (3072, ), (1, ))
    assert_size_stride(primals_1038, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1039, (512, ), (1, ))
    assert_size_stride(primals_1040, (512, ), (1, ))
    assert_size_stride(primals_1041, (512, ), (1, ))
    assert_size_stride(primals_1042, (512, ), (1, ))
    assert_size_stride(primals_1043, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1044, (3072, ), (1, ))
    assert_size_stride(primals_1045, (3072, ), (1, ))
    assert_size_stride(primals_1046, (3072, ), (1, ))
    assert_size_stride(primals_1047, (3072, ), (1, ))
    assert_size_stride(primals_1048, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1049, (3072, ), (1, ))
    assert_size_stride(primals_1050, (3072, ), (1, ))
    assert_size_stride(primals_1051, (3072, ), (1, ))
    assert_size_stride(primals_1052, (3072, ), (1, ))
    assert_size_stride(primals_1053, (128, 3072), (3072, 1))
    assert_size_stride(primals_1054, (128, ), (1, ))
    assert_size_stride(primals_1055, (3072, 128), (128, 1))
    assert_size_stride(primals_1056, (3072, ), (1, ))
    assert_size_stride(primals_1057, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1058, (512, ), (1, ))
    assert_size_stride(primals_1059, (512, ), (1, ))
    assert_size_stride(primals_1060, (512, ), (1, ))
    assert_size_stride(primals_1061, (512, ), (1, ))
    assert_size_stride(primals_1062, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1063, (3072, ), (1, ))
    assert_size_stride(primals_1064, (3072, ), (1, ))
    assert_size_stride(primals_1065, (3072, ), (1, ))
    assert_size_stride(primals_1066, (3072, ), (1, ))
    assert_size_stride(primals_1067, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1068, (3072, ), (1, ))
    assert_size_stride(primals_1069, (3072, ), (1, ))
    assert_size_stride(primals_1070, (3072, ), (1, ))
    assert_size_stride(primals_1071, (3072, ), (1, ))
    assert_size_stride(primals_1072, (128, 3072), (3072, 1))
    assert_size_stride(primals_1073, (128, ), (1, ))
    assert_size_stride(primals_1074, (3072, 128), (128, 1))
    assert_size_stride(primals_1075, (3072, ), (1, ))
    assert_size_stride(primals_1076, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1077, (512, ), (1, ))
    assert_size_stride(primals_1078, (512, ), (1, ))
    assert_size_stride(primals_1079, (512, ), (1, ))
    assert_size_stride(primals_1080, (512, ), (1, ))
    assert_size_stride(primals_1081, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1082, (3072, ), (1, ))
    assert_size_stride(primals_1083, (3072, ), (1, ))
    assert_size_stride(primals_1084, (3072, ), (1, ))
    assert_size_stride(primals_1085, (3072, ), (1, ))
    assert_size_stride(primals_1086, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1087, (3072, ), (1, ))
    assert_size_stride(primals_1088, (3072, ), (1, ))
    assert_size_stride(primals_1089, (3072, ), (1, ))
    assert_size_stride(primals_1090, (3072, ), (1, ))
    assert_size_stride(primals_1091, (128, 3072), (3072, 1))
    assert_size_stride(primals_1092, (128, ), (1, ))
    assert_size_stride(primals_1093, (3072, 128), (128, 1))
    assert_size_stride(primals_1094, (3072, ), (1, ))
    assert_size_stride(primals_1095, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1096, (512, ), (1, ))
    assert_size_stride(primals_1097, (512, ), (1, ))
    assert_size_stride(primals_1098, (512, ), (1, ))
    assert_size_stride(primals_1099, (512, ), (1, ))
    assert_size_stride(primals_1100, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1101, (3072, ), (1, ))
    assert_size_stride(primals_1102, (3072, ), (1, ))
    assert_size_stride(primals_1103, (3072, ), (1, ))
    assert_size_stride(primals_1104, (3072, ), (1, ))
    assert_size_stride(primals_1105, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1106, (3072, ), (1, ))
    assert_size_stride(primals_1107, (3072, ), (1, ))
    assert_size_stride(primals_1108, (3072, ), (1, ))
    assert_size_stride(primals_1109, (3072, ), (1, ))
    assert_size_stride(primals_1110, (128, 3072), (3072, 1))
    assert_size_stride(primals_1111, (128, ), (1, ))
    assert_size_stride(primals_1112, (3072, 128), (128, 1))
    assert_size_stride(primals_1113, (3072, ), (1, ))
    assert_size_stride(primals_1114, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1115, (512, ), (1, ))
    assert_size_stride(primals_1116, (512, ), (1, ))
    assert_size_stride(primals_1117, (512, ), (1, ))
    assert_size_stride(primals_1118, (512, ), (1, ))
    assert_size_stride(primals_1119, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1120, (3072, ), (1, ))
    assert_size_stride(primals_1121, (3072, ), (1, ))
    assert_size_stride(primals_1122, (3072, ), (1, ))
    assert_size_stride(primals_1123, (3072, ), (1, ))
    assert_size_stride(primals_1124, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1125, (3072, ), (1, ))
    assert_size_stride(primals_1126, (3072, ), (1, ))
    assert_size_stride(primals_1127, (3072, ), (1, ))
    assert_size_stride(primals_1128, (3072, ), (1, ))
    assert_size_stride(primals_1129, (128, 3072), (3072, 1))
    assert_size_stride(primals_1130, (128, ), (1, ))
    assert_size_stride(primals_1131, (3072, 128), (128, 1))
    assert_size_stride(primals_1132, (3072, ), (1, ))
    assert_size_stride(primals_1133, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1134, (512, ), (1, ))
    assert_size_stride(primals_1135, (512, ), (1, ))
    assert_size_stride(primals_1136, (512, ), (1, ))
    assert_size_stride(primals_1137, (512, ), (1, ))
    assert_size_stride(primals_1138, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1139, (3072, ), (1, ))
    assert_size_stride(primals_1140, (3072, ), (1, ))
    assert_size_stride(primals_1141, (3072, ), (1, ))
    assert_size_stride(primals_1142, (3072, ), (1, ))
    assert_size_stride(primals_1143, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1144, (3072, ), (1, ))
    assert_size_stride(primals_1145, (3072, ), (1, ))
    assert_size_stride(primals_1146, (3072, ), (1, ))
    assert_size_stride(primals_1147, (3072, ), (1, ))
    assert_size_stride(primals_1148, (128, 3072), (3072, 1))
    assert_size_stride(primals_1149, (128, ), (1, ))
    assert_size_stride(primals_1150, (3072, 128), (128, 1))
    assert_size_stride(primals_1151, (3072, ), (1, ))
    assert_size_stride(primals_1152, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1153, (512, ), (1, ))
    assert_size_stride(primals_1154, (512, ), (1, ))
    assert_size_stride(primals_1155, (512, ), (1, ))
    assert_size_stride(primals_1156, (512, ), (1, ))
    assert_size_stride(primals_1157, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1158, (3072, ), (1, ))
    assert_size_stride(primals_1159, (3072, ), (1, ))
    assert_size_stride(primals_1160, (3072, ), (1, ))
    assert_size_stride(primals_1161, (3072, ), (1, ))
    assert_size_stride(primals_1162, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1163, (3072, ), (1, ))
    assert_size_stride(primals_1164, (3072, ), (1, ))
    assert_size_stride(primals_1165, (3072, ), (1, ))
    assert_size_stride(primals_1166, (3072, ), (1, ))
    assert_size_stride(primals_1167, (128, 3072), (3072, 1))
    assert_size_stride(primals_1168, (128, ), (1, ))
    assert_size_stride(primals_1169, (3072, 128), (128, 1))
    assert_size_stride(primals_1170, (3072, ), (1, ))
    assert_size_stride(primals_1171, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1172, (512, ), (1, ))
    assert_size_stride(primals_1173, (512, ), (1, ))
    assert_size_stride(primals_1174, (512, ), (1, ))
    assert_size_stride(primals_1175, (512, ), (1, ))
    assert_size_stride(primals_1176, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1177, (3072, ), (1, ))
    assert_size_stride(primals_1178, (3072, ), (1, ))
    assert_size_stride(primals_1179, (3072, ), (1, ))
    assert_size_stride(primals_1180, (3072, ), (1, ))
    assert_size_stride(primals_1181, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1182, (3072, ), (1, ))
    assert_size_stride(primals_1183, (3072, ), (1, ))
    assert_size_stride(primals_1184, (3072, ), (1, ))
    assert_size_stride(primals_1185, (3072, ), (1, ))
    assert_size_stride(primals_1186, (128, 3072), (3072, 1))
    assert_size_stride(primals_1187, (128, ), (1, ))
    assert_size_stride(primals_1188, (3072, 128), (128, 1))
    assert_size_stride(primals_1189, (3072, ), (1, ))
    assert_size_stride(primals_1190, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1191, (512, ), (1, ))
    assert_size_stride(primals_1192, (512, ), (1, ))
    assert_size_stride(primals_1193, (512, ), (1, ))
    assert_size_stride(primals_1194, (512, ), (1, ))
    assert_size_stride(primals_1195, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1196, (3072, ), (1, ))
    assert_size_stride(primals_1197, (3072, ), (1, ))
    assert_size_stride(primals_1198, (3072, ), (1, ))
    assert_size_stride(primals_1199, (3072, ), (1, ))
    assert_size_stride(primals_1200, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1201, (3072, ), (1, ))
    assert_size_stride(primals_1202, (3072, ), (1, ))
    assert_size_stride(primals_1203, (3072, ), (1, ))
    assert_size_stride(primals_1204, (3072, ), (1, ))
    assert_size_stride(primals_1205, (128, 3072), (3072, 1))
    assert_size_stride(primals_1206, (128, ), (1, ))
    assert_size_stride(primals_1207, (3072, 128), (128, 1))
    assert_size_stride(primals_1208, (3072, ), (1, ))
    assert_size_stride(primals_1209, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1210, (512, ), (1, ))
    assert_size_stride(primals_1211, (512, ), (1, ))
    assert_size_stride(primals_1212, (512, ), (1, ))
    assert_size_stride(primals_1213, (512, ), (1, ))
    assert_size_stride(primals_1214, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1215, (3072, ), (1, ))
    assert_size_stride(primals_1216, (3072, ), (1, ))
    assert_size_stride(primals_1217, (3072, ), (1, ))
    assert_size_stride(primals_1218, (3072, ), (1, ))
    assert_size_stride(primals_1219, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1220, (3072, ), (1, ))
    assert_size_stride(primals_1221, (3072, ), (1, ))
    assert_size_stride(primals_1222, (3072, ), (1, ))
    assert_size_stride(primals_1223, (3072, ), (1, ))
    assert_size_stride(primals_1224, (128, 3072), (3072, 1))
    assert_size_stride(primals_1225, (128, ), (1, ))
    assert_size_stride(primals_1226, (3072, 128), (128, 1))
    assert_size_stride(primals_1227, (3072, ), (1, ))
    assert_size_stride(primals_1228, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1229, (512, ), (1, ))
    assert_size_stride(primals_1230, (512, ), (1, ))
    assert_size_stride(primals_1231, (512, ), (1, ))
    assert_size_stride(primals_1232, (512, ), (1, ))
    assert_size_stride(primals_1233, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1234, (3072, ), (1, ))
    assert_size_stride(primals_1235, (3072, ), (1, ))
    assert_size_stride(primals_1236, (3072, ), (1, ))
    assert_size_stride(primals_1237, (3072, ), (1, ))
    assert_size_stride(primals_1238, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1239, (3072, ), (1, ))
    assert_size_stride(primals_1240, (3072, ), (1, ))
    assert_size_stride(primals_1241, (3072, ), (1, ))
    assert_size_stride(primals_1242, (3072, ), (1, ))
    assert_size_stride(primals_1243, (128, 3072), (3072, 1))
    assert_size_stride(primals_1244, (128, ), (1, ))
    assert_size_stride(primals_1245, (3072, 128), (128, 1))
    assert_size_stride(primals_1246, (3072, ), (1, ))
    assert_size_stride(primals_1247, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1248, (512, ), (1, ))
    assert_size_stride(primals_1249, (512, ), (1, ))
    assert_size_stride(primals_1250, (512, ), (1, ))
    assert_size_stride(primals_1251, (512, ), (1, ))
    assert_size_stride(primals_1252, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1253, (3072, ), (1, ))
    assert_size_stride(primals_1254, (3072, ), (1, ))
    assert_size_stride(primals_1255, (3072, ), (1, ))
    assert_size_stride(primals_1256, (3072, ), (1, ))
    assert_size_stride(primals_1257, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1258, (3072, ), (1, ))
    assert_size_stride(primals_1259, (3072, ), (1, ))
    assert_size_stride(primals_1260, (3072, ), (1, ))
    assert_size_stride(primals_1261, (3072, ), (1, ))
    assert_size_stride(primals_1262, (128, 3072), (3072, 1))
    assert_size_stride(primals_1263, (128, ), (1, ))
    assert_size_stride(primals_1264, (3072, 128), (128, 1))
    assert_size_stride(primals_1265, (3072, ), (1, ))
    assert_size_stride(primals_1266, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1267, (512, ), (1, ))
    assert_size_stride(primals_1268, (512, ), (1, ))
    assert_size_stride(primals_1269, (512, ), (1, ))
    assert_size_stride(primals_1270, (512, ), (1, ))
    assert_size_stride(primals_1271, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1272, (3072, ), (1, ))
    assert_size_stride(primals_1273, (3072, ), (1, ))
    assert_size_stride(primals_1274, (3072, ), (1, ))
    assert_size_stride(primals_1275, (3072, ), (1, ))
    assert_size_stride(primals_1276, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1277, (3072, ), (1, ))
    assert_size_stride(primals_1278, (3072, ), (1, ))
    assert_size_stride(primals_1279, (3072, ), (1, ))
    assert_size_stride(primals_1280, (3072, ), (1, ))
    assert_size_stride(primals_1281, (128, 3072), (3072, 1))
    assert_size_stride(primals_1282, (128, ), (1, ))
    assert_size_stride(primals_1283, (3072, 128), (128, 1))
    assert_size_stride(primals_1284, (3072, ), (1, ))
    assert_size_stride(primals_1285, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1286, (512, ), (1, ))
    assert_size_stride(primals_1287, (512, ), (1, ))
    assert_size_stride(primals_1288, (512, ), (1, ))
    assert_size_stride(primals_1289, (512, ), (1, ))
    assert_size_stride(primals_1290, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1291, (3072, ), (1, ))
    assert_size_stride(primals_1292, (3072, ), (1, ))
    assert_size_stride(primals_1293, (3072, ), (1, ))
    assert_size_stride(primals_1294, (3072, ), (1, ))
    assert_size_stride(primals_1295, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1296, (3072, ), (1, ))
    assert_size_stride(primals_1297, (3072, ), (1, ))
    assert_size_stride(primals_1298, (3072, ), (1, ))
    assert_size_stride(primals_1299, (3072, ), (1, ))
    assert_size_stride(primals_1300, (128, 3072), (3072, 1))
    assert_size_stride(primals_1301, (128, ), (1, ))
    assert_size_stride(primals_1302, (3072, 128), (128, 1))
    assert_size_stride(primals_1303, (3072, ), (1, ))
    assert_size_stride(primals_1304, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1305, (512, ), (1, ))
    assert_size_stride(primals_1306, (512, ), (1, ))
    assert_size_stride(primals_1307, (512, ), (1, ))
    assert_size_stride(primals_1308, (512, ), (1, ))
    assert_size_stride(primals_1309, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1310, (3072, ), (1, ))
    assert_size_stride(primals_1311, (3072, ), (1, ))
    assert_size_stride(primals_1312, (3072, ), (1, ))
    assert_size_stride(primals_1313, (3072, ), (1, ))
    assert_size_stride(primals_1314, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1315, (3072, ), (1, ))
    assert_size_stride(primals_1316, (3072, ), (1, ))
    assert_size_stride(primals_1317, (3072, ), (1, ))
    assert_size_stride(primals_1318, (3072, ), (1, ))
    assert_size_stride(primals_1319, (128, 3072), (3072, 1))
    assert_size_stride(primals_1320, (128, ), (1, ))
    assert_size_stride(primals_1321, (3072, 128), (128, 1))
    assert_size_stride(primals_1322, (3072, ), (1, ))
    assert_size_stride(primals_1323, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1324, (512, ), (1, ))
    assert_size_stride(primals_1325, (512, ), (1, ))
    assert_size_stride(primals_1326, (512, ), (1, ))
    assert_size_stride(primals_1327, (512, ), (1, ))
    assert_size_stride(primals_1328, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1329, (3072, ), (1, ))
    assert_size_stride(primals_1330, (3072, ), (1, ))
    assert_size_stride(primals_1331, (3072, ), (1, ))
    assert_size_stride(primals_1332, (3072, ), (1, ))
    assert_size_stride(primals_1333, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1334, (3072, ), (1, ))
    assert_size_stride(primals_1335, (3072, ), (1, ))
    assert_size_stride(primals_1336, (3072, ), (1, ))
    assert_size_stride(primals_1337, (3072, ), (1, ))
    assert_size_stride(primals_1338, (128, 3072), (3072, 1))
    assert_size_stride(primals_1339, (128, ), (1, ))
    assert_size_stride(primals_1340, (3072, 128), (128, 1))
    assert_size_stride(primals_1341, (3072, ), (1, ))
    assert_size_stride(primals_1342, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1343, (512, ), (1, ))
    assert_size_stride(primals_1344, (512, ), (1, ))
    assert_size_stride(primals_1345, (512, ), (1, ))
    assert_size_stride(primals_1346, (512, ), (1, ))
    assert_size_stride(primals_1347, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1348, (3072, ), (1, ))
    assert_size_stride(primals_1349, (3072, ), (1, ))
    assert_size_stride(primals_1350, (3072, ), (1, ))
    assert_size_stride(primals_1351, (3072, ), (1, ))
    assert_size_stride(primals_1352, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1353, (3072, ), (1, ))
    assert_size_stride(primals_1354, (3072, ), (1, ))
    assert_size_stride(primals_1355, (3072, ), (1, ))
    assert_size_stride(primals_1356, (3072, ), (1, ))
    assert_size_stride(primals_1357, (128, 3072), (3072, 1))
    assert_size_stride(primals_1358, (128, ), (1, ))
    assert_size_stride(primals_1359, (3072, 128), (128, 1))
    assert_size_stride(primals_1360, (3072, ), (1, ))
    assert_size_stride(primals_1361, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1362, (512, ), (1, ))
    assert_size_stride(primals_1363, (512, ), (1, ))
    assert_size_stride(primals_1364, (512, ), (1, ))
    assert_size_stride(primals_1365, (512, ), (1, ))
    assert_size_stride(primals_1366, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1367, (3072, ), (1, ))
    assert_size_stride(primals_1368, (3072, ), (1, ))
    assert_size_stride(primals_1369, (3072, ), (1, ))
    assert_size_stride(primals_1370, (3072, ), (1, ))
    assert_size_stride(primals_1371, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1372, (3072, ), (1, ))
    assert_size_stride(primals_1373, (3072, ), (1, ))
    assert_size_stride(primals_1374, (3072, ), (1, ))
    assert_size_stride(primals_1375, (3072, ), (1, ))
    assert_size_stride(primals_1376, (128, 3072), (3072, 1))
    assert_size_stride(primals_1377, (128, ), (1, ))
    assert_size_stride(primals_1378, (3072, 128), (128, 1))
    assert_size_stride(primals_1379, (3072, ), (1, ))
    assert_size_stride(primals_1380, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1381, (512, ), (1, ))
    assert_size_stride(primals_1382, (512, ), (1, ))
    assert_size_stride(primals_1383, (512, ), (1, ))
    assert_size_stride(primals_1384, (512, ), (1, ))
    assert_size_stride(primals_1385, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1386, (3072, ), (1, ))
    assert_size_stride(primals_1387, (3072, ), (1, ))
    assert_size_stride(primals_1388, (3072, ), (1, ))
    assert_size_stride(primals_1389, (3072, ), (1, ))
    assert_size_stride(primals_1390, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1391, (3072, ), (1, ))
    assert_size_stride(primals_1392, (3072, ), (1, ))
    assert_size_stride(primals_1393, (3072, ), (1, ))
    assert_size_stride(primals_1394, (3072, ), (1, ))
    assert_size_stride(primals_1395, (128, 3072), (3072, 1))
    assert_size_stride(primals_1396, (128, ), (1, ))
    assert_size_stride(primals_1397, (3072, 128), (128, 1))
    assert_size_stride(primals_1398, (3072, ), (1, ))
    assert_size_stride(primals_1399, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1400, (512, ), (1, ))
    assert_size_stride(primals_1401, (512, ), (1, ))
    assert_size_stride(primals_1402, (512, ), (1, ))
    assert_size_stride(primals_1403, (512, ), (1, ))
    assert_size_stride(primals_1404, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1405, (3072, ), (1, ))
    assert_size_stride(primals_1406, (3072, ), (1, ))
    assert_size_stride(primals_1407, (3072, ), (1, ))
    assert_size_stride(primals_1408, (3072, ), (1, ))
    assert_size_stride(primals_1409, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1410, (3072, ), (1, ))
    assert_size_stride(primals_1411, (3072, ), (1, ))
    assert_size_stride(primals_1412, (3072, ), (1, ))
    assert_size_stride(primals_1413, (3072, ), (1, ))
    assert_size_stride(primals_1414, (128, 3072), (3072, 1))
    assert_size_stride(primals_1415, (128, ), (1, ))
    assert_size_stride(primals_1416, (3072, 128), (128, 1))
    assert_size_stride(primals_1417, (3072, ), (1, ))
    assert_size_stride(primals_1418, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1419, (512, ), (1, ))
    assert_size_stride(primals_1420, (512, ), (1, ))
    assert_size_stride(primals_1421, (512, ), (1, ))
    assert_size_stride(primals_1422, (512, ), (1, ))
    assert_size_stride(primals_1423, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1424, (3072, ), (1, ))
    assert_size_stride(primals_1425, (3072, ), (1, ))
    assert_size_stride(primals_1426, (3072, ), (1, ))
    assert_size_stride(primals_1427, (3072, ), (1, ))
    assert_size_stride(primals_1428, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1429, (3072, ), (1, ))
    assert_size_stride(primals_1430, (3072, ), (1, ))
    assert_size_stride(primals_1431, (3072, ), (1, ))
    assert_size_stride(primals_1432, (3072, ), (1, ))
    assert_size_stride(primals_1433, (128, 3072), (3072, 1))
    assert_size_stride(primals_1434, (128, ), (1, ))
    assert_size_stride(primals_1435, (3072, 128), (128, 1))
    assert_size_stride(primals_1436, (3072, ), (1, ))
    assert_size_stride(primals_1437, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1438, (512, ), (1, ))
    assert_size_stride(primals_1439, (512, ), (1, ))
    assert_size_stride(primals_1440, (512, ), (1, ))
    assert_size_stride(primals_1441, (512, ), (1, ))
    assert_size_stride(primals_1442, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1443, (3072, ), (1, ))
    assert_size_stride(primals_1444, (3072, ), (1, ))
    assert_size_stride(primals_1445, (3072, ), (1, ))
    assert_size_stride(primals_1446, (3072, ), (1, ))
    assert_size_stride(primals_1447, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1448, (3072, ), (1, ))
    assert_size_stride(primals_1449, (3072, ), (1, ))
    assert_size_stride(primals_1450, (3072, ), (1, ))
    assert_size_stride(primals_1451, (3072, ), (1, ))
    assert_size_stride(primals_1452, (128, 3072), (3072, 1))
    assert_size_stride(primals_1453, (128, ), (1, ))
    assert_size_stride(primals_1454, (3072, 128), (128, 1))
    assert_size_stride(primals_1455, (3072, ), (1, ))
    assert_size_stride(primals_1456, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1457, (512, ), (1, ))
    assert_size_stride(primals_1458, (512, ), (1, ))
    assert_size_stride(primals_1459, (512, ), (1, ))
    assert_size_stride(primals_1460, (512, ), (1, ))
    assert_size_stride(primals_1461, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1462, (3072, ), (1, ))
    assert_size_stride(primals_1463, (3072, ), (1, ))
    assert_size_stride(primals_1464, (3072, ), (1, ))
    assert_size_stride(primals_1465, (3072, ), (1, ))
    assert_size_stride(primals_1466, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1467, (3072, ), (1, ))
    assert_size_stride(primals_1468, (3072, ), (1, ))
    assert_size_stride(primals_1469, (3072, ), (1, ))
    assert_size_stride(primals_1470, (3072, ), (1, ))
    assert_size_stride(primals_1471, (128, 3072), (3072, 1))
    assert_size_stride(primals_1472, (128, ), (1, ))
    assert_size_stride(primals_1473, (3072, 128), (128, 1))
    assert_size_stride(primals_1474, (3072, ), (1, ))
    assert_size_stride(primals_1475, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1476, (512, ), (1, ))
    assert_size_stride(primals_1477, (512, ), (1, ))
    assert_size_stride(primals_1478, (512, ), (1, ))
    assert_size_stride(primals_1479, (512, ), (1, ))
    assert_size_stride(primals_1480, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1481, (3072, ), (1, ))
    assert_size_stride(primals_1482, (3072, ), (1, ))
    assert_size_stride(primals_1483, (3072, ), (1, ))
    assert_size_stride(primals_1484, (3072, ), (1, ))
    assert_size_stride(primals_1485, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1486, (3072, ), (1, ))
    assert_size_stride(primals_1487, (3072, ), (1, ))
    assert_size_stride(primals_1488, (3072, ), (1, ))
    assert_size_stride(primals_1489, (3072, ), (1, ))
    assert_size_stride(primals_1490, (128, 3072), (3072, 1))
    assert_size_stride(primals_1491, (128, ), (1, ))
    assert_size_stride(primals_1492, (3072, 128), (128, 1))
    assert_size_stride(primals_1493, (3072, ), (1, ))
    assert_size_stride(primals_1494, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1495, (512, ), (1, ))
    assert_size_stride(primals_1496, (512, ), (1, ))
    assert_size_stride(primals_1497, (512, ), (1, ))
    assert_size_stride(primals_1498, (512, ), (1, ))
    assert_size_stride(primals_1499, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1500, (3072, ), (1, ))
    assert_size_stride(primals_1501, (3072, ), (1, ))
    assert_size_stride(primals_1502, (3072, ), (1, ))
    assert_size_stride(primals_1503, (3072, ), (1, ))
    assert_size_stride(primals_1504, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1505, (3072, ), (1, ))
    assert_size_stride(primals_1506, (3072, ), (1, ))
    assert_size_stride(primals_1507, (3072, ), (1, ))
    assert_size_stride(primals_1508, (3072, ), (1, ))
    assert_size_stride(primals_1509, (128, 3072), (3072, 1))
    assert_size_stride(primals_1510, (128, ), (1, ))
    assert_size_stride(primals_1511, (3072, 128), (128, 1))
    assert_size_stride(primals_1512, (3072, ), (1, ))
    assert_size_stride(primals_1513, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1514, (512, ), (1, ))
    assert_size_stride(primals_1515, (512, ), (1, ))
    assert_size_stride(primals_1516, (512, ), (1, ))
    assert_size_stride(primals_1517, (512, ), (1, ))
    assert_size_stride(primals_1518, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1519, (3072, ), (1, ))
    assert_size_stride(primals_1520, (3072, ), (1, ))
    assert_size_stride(primals_1521, (3072, ), (1, ))
    assert_size_stride(primals_1522, (3072, ), (1, ))
    assert_size_stride(primals_1523, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1524, (3072, ), (1, ))
    assert_size_stride(primals_1525, (3072, ), (1, ))
    assert_size_stride(primals_1526, (3072, ), (1, ))
    assert_size_stride(primals_1527, (3072, ), (1, ))
    assert_size_stride(primals_1528, (128, 3072), (3072, 1))
    assert_size_stride(primals_1529, (128, ), (1, ))
    assert_size_stride(primals_1530, (3072, 128), (128, 1))
    assert_size_stride(primals_1531, (3072, ), (1, ))
    assert_size_stride(primals_1532, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1533, (512, ), (1, ))
    assert_size_stride(primals_1534, (512, ), (1, ))
    assert_size_stride(primals_1535, (512, ), (1, ))
    assert_size_stride(primals_1536, (512, ), (1, ))
    assert_size_stride(primals_1537, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1538, (3072, ), (1, ))
    assert_size_stride(primals_1539, (3072, ), (1, ))
    assert_size_stride(primals_1540, (3072, ), (1, ))
    assert_size_stride(primals_1541, (3072, ), (1, ))
    assert_size_stride(primals_1542, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1543, (3072, ), (1, ))
    assert_size_stride(primals_1544, (3072, ), (1, ))
    assert_size_stride(primals_1545, (3072, ), (1, ))
    assert_size_stride(primals_1546, (3072, ), (1, ))
    assert_size_stride(primals_1547, (128, 3072), (3072, 1))
    assert_size_stride(primals_1548, (128, ), (1, ))
    assert_size_stride(primals_1549, (3072, 128), (128, 1))
    assert_size_stride(primals_1550, (3072, ), (1, ))
    assert_size_stride(primals_1551, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1552, (512, ), (1, ))
    assert_size_stride(primals_1553, (512, ), (1, ))
    assert_size_stride(primals_1554, (512, ), (1, ))
    assert_size_stride(primals_1555, (512, ), (1, ))
    assert_size_stride(primals_1556, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1557, (3072, ), (1, ))
    assert_size_stride(primals_1558, (3072, ), (1, ))
    assert_size_stride(primals_1559, (3072, ), (1, ))
    assert_size_stride(primals_1560, (3072, ), (1, ))
    assert_size_stride(primals_1561, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1562, (3072, ), (1, ))
    assert_size_stride(primals_1563, (3072, ), (1, ))
    assert_size_stride(primals_1564, (3072, ), (1, ))
    assert_size_stride(primals_1565, (3072, ), (1, ))
    assert_size_stride(primals_1566, (128, 3072), (3072, 1))
    assert_size_stride(primals_1567, (128, ), (1, ))
    assert_size_stride(primals_1568, (3072, 128), (128, 1))
    assert_size_stride(primals_1569, (3072, ), (1, ))
    assert_size_stride(primals_1570, (512, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1571, (512, ), (1, ))
    assert_size_stride(primals_1572, (512, ), (1, ))
    assert_size_stride(primals_1573, (512, ), (1, ))
    assert_size_stride(primals_1574, (512, ), (1, ))
    assert_size_stride(primals_1575, (3072, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_1576, (3072, ), (1, ))
    assert_size_stride(primals_1577, (3072, ), (1, ))
    assert_size_stride(primals_1578, (3072, ), (1, ))
    assert_size_stride(primals_1579, (3072, ), (1, ))
    assert_size_stride(primals_1580, (3072, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1581, (3072, ), (1, ))
    assert_size_stride(primals_1582, (3072, ), (1, ))
    assert_size_stride(primals_1583, (3072, ), (1, ))
    assert_size_stride(primals_1584, (3072, ), (1, ))
    assert_size_stride(primals_1585, (128, 3072), (3072, 1))
    assert_size_stride(primals_1586, (128, ), (1, ))
    assert_size_stride(primals_1587, (3072, 128), (128, 1))
    assert_size_stride(primals_1588, (3072, ), (1, ))
    assert_size_stride(primals_1589, (640, 3072, 1, 1), (3072, 1, 1, 1))
    assert_size_stride(primals_1590, (640, ), (1, ))
    assert_size_stride(primals_1591, (640, ), (1, ))
    assert_size_stride(primals_1592, (640, ), (1, ))
    assert_size_stride(primals_1593, (640, ), (1, ))
    assert_size_stride(primals_1594, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1595, (3840, ), (1, ))
    assert_size_stride(primals_1596, (3840, ), (1, ))
    assert_size_stride(primals_1597, (3840, ), (1, ))
    assert_size_stride(primals_1598, (3840, ), (1, ))
    assert_size_stride(primals_1599, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1600, (3840, ), (1, ))
    assert_size_stride(primals_1601, (3840, ), (1, ))
    assert_size_stride(primals_1602, (3840, ), (1, ))
    assert_size_stride(primals_1603, (3840, ), (1, ))
    assert_size_stride(primals_1604, (160, 3840), (3840, 1))
    assert_size_stride(primals_1605, (160, ), (1, ))
    assert_size_stride(primals_1606, (3840, 160), (160, 1))
    assert_size_stride(primals_1607, (3840, ), (1, ))
    assert_size_stride(primals_1608, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1609, (640, ), (1, ))
    assert_size_stride(primals_1610, (640, ), (1, ))
    assert_size_stride(primals_1611, (640, ), (1, ))
    assert_size_stride(primals_1612, (640, ), (1, ))
    assert_size_stride(primals_1613, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1614, (3840, ), (1, ))
    assert_size_stride(primals_1615, (3840, ), (1, ))
    assert_size_stride(primals_1616, (3840, ), (1, ))
    assert_size_stride(primals_1617, (3840, ), (1, ))
    assert_size_stride(primals_1618, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1619, (3840, ), (1, ))
    assert_size_stride(primals_1620, (3840, ), (1, ))
    assert_size_stride(primals_1621, (3840, ), (1, ))
    assert_size_stride(primals_1622, (3840, ), (1, ))
    assert_size_stride(primals_1623, (160, 3840), (3840, 1))
    assert_size_stride(primals_1624, (160, ), (1, ))
    assert_size_stride(primals_1625, (3840, 160), (160, 1))
    assert_size_stride(primals_1626, (3840, ), (1, ))
    assert_size_stride(primals_1627, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1628, (640, ), (1, ))
    assert_size_stride(primals_1629, (640, ), (1, ))
    assert_size_stride(primals_1630, (640, ), (1, ))
    assert_size_stride(primals_1631, (640, ), (1, ))
    assert_size_stride(primals_1632, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1633, (3840, ), (1, ))
    assert_size_stride(primals_1634, (3840, ), (1, ))
    assert_size_stride(primals_1635, (3840, ), (1, ))
    assert_size_stride(primals_1636, (3840, ), (1, ))
    assert_size_stride(primals_1637, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1638, (3840, ), (1, ))
    assert_size_stride(primals_1639, (3840, ), (1, ))
    assert_size_stride(primals_1640, (3840, ), (1, ))
    assert_size_stride(primals_1641, (3840, ), (1, ))
    assert_size_stride(primals_1642, (160, 3840), (3840, 1))
    assert_size_stride(primals_1643, (160, ), (1, ))
    assert_size_stride(primals_1644, (3840, 160), (160, 1))
    assert_size_stride(primals_1645, (3840, ), (1, ))
    assert_size_stride(primals_1646, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1647, (640, ), (1, ))
    assert_size_stride(primals_1648, (640, ), (1, ))
    assert_size_stride(primals_1649, (640, ), (1, ))
    assert_size_stride(primals_1650, (640, ), (1, ))
    assert_size_stride(primals_1651, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1652, (3840, ), (1, ))
    assert_size_stride(primals_1653, (3840, ), (1, ))
    assert_size_stride(primals_1654, (3840, ), (1, ))
    assert_size_stride(primals_1655, (3840, ), (1, ))
    assert_size_stride(primals_1656, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1657, (3840, ), (1, ))
    assert_size_stride(primals_1658, (3840, ), (1, ))
    assert_size_stride(primals_1659, (3840, ), (1, ))
    assert_size_stride(primals_1660, (3840, ), (1, ))
    assert_size_stride(primals_1661, (160, 3840), (3840, 1))
    assert_size_stride(primals_1662, (160, ), (1, ))
    assert_size_stride(primals_1663, (3840, 160), (160, 1))
    assert_size_stride(primals_1664, (3840, ), (1, ))
    assert_size_stride(primals_1665, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1666, (640, ), (1, ))
    assert_size_stride(primals_1667, (640, ), (1, ))
    assert_size_stride(primals_1668, (640, ), (1, ))
    assert_size_stride(primals_1669, (640, ), (1, ))
    assert_size_stride(primals_1670, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1671, (3840, ), (1, ))
    assert_size_stride(primals_1672, (3840, ), (1, ))
    assert_size_stride(primals_1673, (3840, ), (1, ))
    assert_size_stride(primals_1674, (3840, ), (1, ))
    assert_size_stride(primals_1675, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1676, (3840, ), (1, ))
    assert_size_stride(primals_1677, (3840, ), (1, ))
    assert_size_stride(primals_1678, (3840, ), (1, ))
    assert_size_stride(primals_1679, (3840, ), (1, ))
    assert_size_stride(primals_1680, (160, 3840), (3840, 1))
    assert_size_stride(primals_1681, (160, ), (1, ))
    assert_size_stride(primals_1682, (3840, 160), (160, 1))
    assert_size_stride(primals_1683, (3840, ), (1, ))
    assert_size_stride(primals_1684, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1685, (640, ), (1, ))
    assert_size_stride(primals_1686, (640, ), (1, ))
    assert_size_stride(primals_1687, (640, ), (1, ))
    assert_size_stride(primals_1688, (640, ), (1, ))
    assert_size_stride(primals_1689, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1690, (3840, ), (1, ))
    assert_size_stride(primals_1691, (3840, ), (1, ))
    assert_size_stride(primals_1692, (3840, ), (1, ))
    assert_size_stride(primals_1693, (3840, ), (1, ))
    assert_size_stride(primals_1694, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1695, (3840, ), (1, ))
    assert_size_stride(primals_1696, (3840, ), (1, ))
    assert_size_stride(primals_1697, (3840, ), (1, ))
    assert_size_stride(primals_1698, (3840, ), (1, ))
    assert_size_stride(primals_1699, (160, 3840), (3840, 1))
    assert_size_stride(primals_1700, (160, ), (1, ))
    assert_size_stride(primals_1701, (3840, 160), (160, 1))
    assert_size_stride(primals_1702, (3840, ), (1, ))
    assert_size_stride(primals_1703, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1704, (640, ), (1, ))
    assert_size_stride(primals_1705, (640, ), (1, ))
    assert_size_stride(primals_1706, (640, ), (1, ))
    assert_size_stride(primals_1707, (640, ), (1, ))
    assert_size_stride(primals_1708, (3840, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1709, (3840, ), (1, ))
    assert_size_stride(primals_1710, (3840, ), (1, ))
    assert_size_stride(primals_1711, (3840, ), (1, ))
    assert_size_stride(primals_1712, (3840, ), (1, ))
    assert_size_stride(primals_1713, (3840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_1714, (3840, ), (1, ))
    assert_size_stride(primals_1715, (3840, ), (1, ))
    assert_size_stride(primals_1716, (3840, ), (1, ))
    assert_size_stride(primals_1717, (3840, ), (1, ))
    assert_size_stride(primals_1718, (160, 3840), (3840, 1))
    assert_size_stride(primals_1719, (160, ), (1, ))
    assert_size_stride(primals_1720, (3840, 160), (160, 1))
    assert_size_stride(primals_1721, (3840, ), (1, ))
    assert_size_stride(primals_1722, (640, 3840, 1, 1), (3840, 1, 1, 1))
    assert_size_stride(primals_1723, (640, ), (1, ))
    assert_size_stride(primals_1724, (640, ), (1, ))
    assert_size_stride(primals_1725, (640, ), (1, ))
    assert_size_stride(primals_1726, (640, ), (1, ))
    assert_size_stride(primals_1727, (1792, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_1728, (1792, ), (1, ))
    assert_size_stride(primals_1729, (1792, ), (1, ))
    assert_size_stride(primals_1730, (1792, ), (1, ))
    assert_size_stride(primals_1731, (1792, ), (1, ))
    assert_size_stride(primals_1732, (1000, 1792), (1792, 1))
    assert_size_stride(primals_1733, (1000, ), (1, ))
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
        buf14 = empty_strided_cuda((256, 64, 3, 3), (576, 1, 192, 64), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_127, buf14, 16384, 9, grid=grid(16384, 9), stream=stream0)
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
        buf20 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_187, buf20, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_187
        buf21 = empty_strided_cuda((384, 96, 3, 3), (864, 1, 288, 96), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_197, buf21, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del primals_197
        # Topologically Sorted Source Nodes: [features_0_0], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf23 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf24 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [features_0_1, sigmoid_1, mul_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7.run(buf24, buf22, primals_3, primals_4, primals_5, primals_6, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_0], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf26 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        buf27 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [features_1_conv_1, sigmoid_2, mul_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_7.run(buf27, buf25, primals_8, primals_9, primals_10, primals_11, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_1_conv_3], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf29 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_1_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf28, primals_13, primals_14, primals_15, primals_16, buf29, 131072, grid=grid(131072), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [features_2_conv_0], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf31 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf32 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [features_2_conv_1, sigmoid_3, mul_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9.run(buf32, buf30, primals_18, primals_19, primals_20, primals_21, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_2_conv_3], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf34 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_2_conv_4, add_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf29, buf33, primals_23, primals_24, primals_25, primals_26, buf34, 131072, grid=grid(131072), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [features_3_conv_0], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf36 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf37 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [features_3_conv_1, sigmoid_4, mul_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9.run(buf37, buf35, primals_28, primals_29, primals_30, primals_31, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_3_conv_3], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf39 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_3_conv_4, add_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf34, buf38, primals_33, primals_34, primals_35, primals_36, buf39, 131072, grid=grid(131072), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [features_4_conv_0], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf41 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        buf42 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [features_4_conv_1, sigmoid_5, mul_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_9.run(buf42, buf40, primals_38, primals_39, primals_40, primals_41, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_4_conv_3], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf44 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [features_4_conv_4, add_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_10.run(buf39, buf43, primals_43, primals_44, primals_45, primals_46, buf44, 131072, grid=grid(131072), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [features_5_conv_0], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, buf6, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf46 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf47 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [features_5_conv_1, sigmoid_6, mul_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_11.run(buf47, buf45, primals_48, primals_49, primals_50, primals_51, 131072, grid=grid(131072), stream=stream0)
        # Topologically Sorted Source Nodes: [features_5_conv_3], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf49 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_5_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf48, primals_53, primals_54, primals_55, primals_56, buf49, 65536, grid=grid(65536), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [features_6_conv_0], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf51 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf52 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [features_6_conv_1, sigmoid_7, mul_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf52, buf50, primals_58, primals_59, primals_60, primals_61, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_6_conv_3], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf54 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_6_conv_4, add_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf49, buf53, primals_63, primals_64, primals_65, primals_66, buf54, 65536, grid=grid(65536), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [features_7_conv_0], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf56 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf57 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [features_7_conv_1, sigmoid_8, mul_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf57, buf55, primals_68, primals_69, primals_70, primals_71, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_7_conv_3], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf59 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_7_conv_4, add_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf54, buf58, primals_73, primals_74, primals_75, primals_76, buf59, 65536, grid=grid(65536), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [features_8_conv_0], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf61 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [features_8_conv_1, sigmoid_9, mul_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf62, buf60, primals_78, primals_79, primals_80, primals_81, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_8_conv_3], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf64 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_8_conv_4, add_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf59, buf63, primals_83, primals_84, primals_85, primals_86, buf64, 65536, grid=grid(65536), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [features_9_conv_0], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf66 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf67 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [features_9_conv_1, sigmoid_10, mul_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf67, buf65, primals_88, primals_89, primals_90, primals_91, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_9_conv_3], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf69 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_9_conv_4, add_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf64, buf68, primals_93, primals_94, primals_95, primals_96, buf69, 65536, grid=grid(65536), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [features_10_conv_0], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf71 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf72 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [features_10_conv_1, sigmoid_11, mul_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf72, buf70, primals_98, primals_99, primals_100, primals_101, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_10_conv_3], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf74 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_10_conv_4, add_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf69, buf73, primals_103, primals_104, primals_105, primals_106, buf74, 65536, grid=grid(65536), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [features_11_conv_0], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf76 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf77 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [features_11_conv_1, sigmoid_12, mul_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf77, buf75, primals_108, primals_109, primals_110, primals_111, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_11_conv_3], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf79 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_11_conv_4, add_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf74, buf78, primals_113, primals_114, primals_115, primals_116, buf79, 65536, grid=grid(65536), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [features_12_conv_0], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf81 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        buf82 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [features_12_conv_1, sigmoid_13, mul_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_13.run(buf82, buf80, primals_118, primals_119, primals_120, primals_121, 262144, grid=grid(262144), stream=stream0)
        # Topologically Sorted Source Nodes: [features_12_conv_3], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 64, 16, 16), (16384, 1, 1024, 64))
        buf84 = empty_strided_cuda((4, 64, 16, 16), (16384, 1, 1024, 64), torch.float32)
        # Topologically Sorted Source Nodes: [features_12_conv_4, add_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf79, buf83, primals_123, primals_124, primals_125, primals_126, buf84, 65536, grid=grid(65536), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [features_13_conv_0], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, buf14, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf86 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [features_13_conv_1, sigmoid_14, mul_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_15.run(buf87, buf85, primals_128, primals_129, primals_130, primals_131, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [features_13_conv_3], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf89 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_13_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf88, primals_133, primals_134, primals_135, primals_136, buf89, 24576, grid=grid(24576), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [features_14_conv_0], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf91 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf92 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [features_14_conv_1, sigmoid_15, mul_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf92, buf90, primals_138, primals_139, primals_140, primals_141, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_14_conv_3], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf94 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_14_conv_4, add_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf89, buf93, primals_143, primals_144, primals_145, primals_146, buf94, 24576, grid=grid(24576), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [features_15_conv_0], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf96 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [features_15_conv_1, sigmoid_16, mul_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf97, buf95, primals_148, primals_149, primals_150, primals_151, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_15_conv_3], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf99 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_15_conv_4, add_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf94, buf98, primals_153, primals_154, primals_155, primals_156, buf99, 24576, grid=grid(24576), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [features_16_conv_0], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf101 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf102 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [features_16_conv_1, sigmoid_17, mul_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf102, buf100, primals_158, primals_159, primals_160, primals_161, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_16_conv_3], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf104 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_16_conv_4, add_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf99, buf103, primals_163, primals_164, primals_165, primals_166, buf104, 24576, grid=grid(24576), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [features_17_conv_0], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf106 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf107 = buf106; del buf106  # reuse
        # Topologically Sorted Source Nodes: [features_17_conv_1, sigmoid_18, mul_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf107, buf105, primals_168, primals_169, primals_170, primals_171, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_17_conv_3], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf109 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_17_conv_4, add_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf104, buf108, primals_173, primals_174, primals_175, primals_176, buf109, 24576, grid=grid(24576), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [features_18_conv_0], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf111 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf112 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [features_18_conv_1, sigmoid_19, mul_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf112, buf110, primals_178, primals_179, primals_180, primals_181, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_18_conv_3], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf114 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_18_conv_4, add_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf109, buf113, primals_183, primals_184, primals_185, primals_186, buf114, 24576, grid=grid(24576), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [features_19_conv_0], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf116 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf117 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [features_19_conv_1, sigmoid_20, mul_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf117, buf115, primals_188, primals_189, primals_190, primals_191, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_19_conv_3], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf119 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_19_conv_4, add_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf114, buf118, primals_193, primals_194, primals_195, primals_196, buf119, 24576, grid=grid(24576), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [features_20_conv_0], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf121 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf122 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [features_20_conv_1, sigmoid_21, mul_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf122, buf120, primals_198, primals_199, primals_200, primals_201, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_20_conv_3], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 96, 8, 8), (6144, 1, 768, 96))
        buf124 = empty_strided_cuda((4, 96, 8, 8), (6144, 1, 768, 96), torch.float32)
        # Topologically Sorted Source Nodes: [features_20_conv_4, add_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf119, buf123, primals_203, primals_204, primals_205, primals_206, buf124, 24576, grid=grid(24576), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [features_21_conv_0], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 384, 8, 8), (24576, 1, 3072, 384))
        buf126 = empty_strided_cuda((4, 384, 8, 8), (24576, 1, 3072, 384), torch.float32)
        buf127 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [features_21_conv_1, sigmoid_22, mul_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_17.run(buf127, buf125, primals_208, primals_209, primals_210, primals_211, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_3], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_212, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf128, (4, 384, 4, 4), (6144, 1, 1536, 384))
        buf129 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_19.run(buf128, primals_213, primals_214, primals_215, primals_216, buf129, 24576, grid=grid(24576), stream=stream0)
        buf130 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 1536, 1536), torch.float32)
        buf131 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_23, mul_23, features_21_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_20.run(buf131, buf129, 1536, 16, grid=grid(1536), stream=stream0)
        buf132 = empty_strided_cuda((4, 24), (24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_218, reinterpret_tensor(buf131, (4, 384), (384, 1), 0), reinterpret_tensor(primals_217, (384, 24), (1, 384), 0), alpha=1, beta=1, out=buf132)
        del primals_218
        buf133 = empty_strided_cuda((4, 24), (24, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_24, mul_24], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_21.run(buf132, buf133, 96, grid=grid(96), stream=stream0)
        buf134 = empty_strided_cuda((4, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_220, buf133, reinterpret_tensor(primals_219, (24, 384), (1, 24), 0), alpha=1, beta=1, out=buf134)
        del primals_220
        buf135 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_23, mul_23, mul_25], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_22.run(buf135, buf134, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_21_conv_7], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf137 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_21_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_23.run(buf136, primals_222, primals_223, primals_224, primals_225, buf137, 12288, grid=grid(12288), stream=stream0)
        del primals_225
        # Topologically Sorted Source Nodes: [features_22_conv_0], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf137, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf139 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf140 = buf139; del buf139  # reuse
        # Topologically Sorted Source Nodes: [features_22_conv_1, sigmoid_27, mul_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf140, buf138, primals_227, primals_228, primals_229, primals_230, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_3], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf141, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf142 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf141, primals_232, primals_233, primals_234, primals_235, buf142, 49152, grid=grid(49152), stream=stream0)
        buf143 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf144 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_29, mul_27, features_22_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf144, buf142, 3072, 16, grid=grid(3072), stream=stream0)
        buf145 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_237, reinterpret_tensor(buf144, (4, 768), (768, 1), 0), reinterpret_tensor(primals_236, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf145)
        del primals_237
        buf146 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_31, mul_28], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf145, buf146, 192, grid=grid(192), stream=stream0)
        buf147 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_239, buf146, reinterpret_tensor(primals_238, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf147)
        del primals_239
        buf148 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_29, mul_27, mul_29], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf148, buf147, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_22_conv_7], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf150 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_22_conv_8, add_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf137, buf149, primals_241, primals_242, primals_243, primals_244, buf150, 12288, grid=grid(12288), stream=stream0)
        del primals_244
        # Topologically Sorted Source Nodes: [features_23_conv_0], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_245, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf152 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf153 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [features_23_conv_1, sigmoid_34, mul_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf153, buf151, primals_246, primals_247, primals_248, primals_249, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_3], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf154, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf155 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf154, primals_251, primals_252, primals_253, primals_254, buf155, 49152, grid=grid(49152), stream=stream0)
        buf156 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf157 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_36, mul_31, features_23_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf157, buf155, 3072, 16, grid=grid(3072), stream=stream0)
        buf158 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_256, reinterpret_tensor(buf157, (4, 768), (768, 1), 0), reinterpret_tensor(primals_255, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf158)
        del primals_256
        buf159 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_38, mul_32], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf158, buf159, 192, grid=grid(192), stream=stream0)
        buf160 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_258, buf159, reinterpret_tensor(primals_257, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf160)
        del primals_258
        buf161 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_36, mul_31, mul_33], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf161, buf160, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_23_conv_7], Original ATen: [aten.convolution]
        buf162 = extern_kernels.convolution(buf161, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf162, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf163 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_23_conv_8, add_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf150, buf162, primals_260, primals_261, primals_262, primals_263, buf163, 12288, grid=grid(12288), stream=stream0)
        del primals_263
        # Topologically Sorted Source Nodes: [features_24_conv_0], Original ATen: [aten.convolution]
        buf164 = extern_kernels.convolution(buf163, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf165 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf166 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [features_24_conv_1, sigmoid_41, mul_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf166, buf164, primals_265, primals_266, primals_267, primals_268, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_3], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_269, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf167, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf168 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf167, primals_270, primals_271, primals_272, primals_273, buf168, 49152, grid=grid(49152), stream=stream0)
        buf169 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf170 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_43, mul_35, features_24_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf170, buf168, 3072, 16, grid=grid(3072), stream=stream0)
        buf171 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_275, reinterpret_tensor(buf170, (4, 768), (768, 1), 0), reinterpret_tensor(primals_274, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf171)
        del primals_275
        buf172 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_45, mul_36], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf171, buf172, 192, grid=grid(192), stream=stream0)
        buf173 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_277, buf172, reinterpret_tensor(primals_276, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf173)
        del primals_277
        buf174 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_43, mul_35, mul_37], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf174, buf173, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_24_conv_7], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf176 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_24_conv_8, add_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf163, buf175, primals_279, primals_280, primals_281, primals_282, buf176, 12288, grid=grid(12288), stream=stream0)
        del primals_282
        # Topologically Sorted Source Nodes: [features_25_conv_0], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf178 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf179 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [features_25_conv_1, sigmoid_48, mul_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf179, buf177, primals_284, primals_285, primals_286, primals_287, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_3], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf180, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf181 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf180, primals_289, primals_290, primals_291, primals_292, buf181, 49152, grid=grid(49152), stream=stream0)
        buf182 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf183 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_50, mul_39, features_25_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf183, buf181, 3072, 16, grid=grid(3072), stream=stream0)
        buf184 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_294, reinterpret_tensor(buf183, (4, 768), (768, 1), 0), reinterpret_tensor(primals_293, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf184)
        del primals_294
        buf185 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_52, mul_40], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf184, buf185, 192, grid=grid(192), stream=stream0)
        buf186 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_296, buf185, reinterpret_tensor(primals_295, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf186)
        del primals_296
        buf187 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_50, mul_39, mul_41], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf187, buf186, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_25_conv_7], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf189 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_25_conv_8, add_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf176, buf188, primals_298, primals_299, primals_300, primals_301, buf189, 12288, grid=grid(12288), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [features_26_conv_0], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf191 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf192 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [features_26_conv_1, sigmoid_55, mul_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf192, buf190, primals_303, primals_304, primals_305, primals_306, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_3], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf193, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf194 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf193, primals_308, primals_309, primals_310, primals_311, buf194, 49152, grid=grid(49152), stream=stream0)
        buf195 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf196 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_57, mul_43, features_26_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf196, buf194, 3072, 16, grid=grid(3072), stream=stream0)
        buf197 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_313, reinterpret_tensor(buf196, (4, 768), (768, 1), 0), reinterpret_tensor(primals_312, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf197)
        del primals_313
        buf198 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_59, mul_44], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf197, buf198, 192, grid=grid(192), stream=stream0)
        buf199 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_315, buf198, reinterpret_tensor(primals_314, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf199)
        del primals_315
        buf200 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_57, mul_43, mul_45], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf200, buf199, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_26_conv_7], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf202 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_26_conv_8, add_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf189, buf201, primals_317, primals_318, primals_319, primals_320, buf202, 12288, grid=grid(12288), stream=stream0)
        del primals_320
        # Topologically Sorted Source Nodes: [features_27_conv_0], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf204 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf205 = buf204; del buf204  # reuse
        # Topologically Sorted Source Nodes: [features_27_conv_1, sigmoid_62, mul_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf205, buf203, primals_322, primals_323, primals_324, primals_325, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_3], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_326, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf206, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf207 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf206, primals_327, primals_328, primals_329, primals_330, buf207, 49152, grid=grid(49152), stream=stream0)
        buf208 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf209 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_64, mul_47, features_27_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf209, buf207, 3072, 16, grid=grid(3072), stream=stream0)
        buf210 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_332, reinterpret_tensor(buf209, (4, 768), (768, 1), 0), reinterpret_tensor(primals_331, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf210)
        del primals_332
        buf211 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_66, mul_48], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf210, buf211, 192, grid=grid(192), stream=stream0)
        buf212 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_334, buf211, reinterpret_tensor(primals_333, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf212)
        del primals_334
        buf213 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_64, mul_47, mul_49], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf213, buf212, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_27_conv_7], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf215 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_27_conv_8, add_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf202, buf214, primals_336, primals_337, primals_338, primals_339, buf215, 12288, grid=grid(12288), stream=stream0)
        del primals_339
        # Topologically Sorted Source Nodes: [features_28_conv_0], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_340, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf217 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf218 = buf217; del buf217  # reuse
        # Topologically Sorted Source Nodes: [features_28_conv_1, sigmoid_69, mul_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf218, buf216, primals_341, primals_342, primals_343, primals_344, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_3], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf219, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf220 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf219, primals_346, primals_347, primals_348, primals_349, buf220, 49152, grid=grid(49152), stream=stream0)
        buf221 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf222 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_71, mul_51, features_28_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf222, buf220, 3072, 16, grid=grid(3072), stream=stream0)
        buf223 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_351, reinterpret_tensor(buf222, (4, 768), (768, 1), 0), reinterpret_tensor(primals_350, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf223)
        del primals_351
        buf224 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_73, mul_52], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf223, buf224, 192, grid=grid(192), stream=stream0)
        buf225 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_353, buf224, reinterpret_tensor(primals_352, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf225)
        del primals_353
        buf226 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_71, mul_51, mul_53], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf226, buf225, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_28_conv_7], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_354, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf228 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_28_conv_8, add_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf215, buf227, primals_355, primals_356, primals_357, primals_358, buf228, 12288, grid=grid(12288), stream=stream0)
        del primals_358
        # Topologically Sorted Source Nodes: [features_29_conv_0], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_359, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf230 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf231 = buf230; del buf230  # reuse
        # Topologically Sorted Source Nodes: [features_29_conv_1, sigmoid_76, mul_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf231, buf229, primals_360, primals_361, primals_362, primals_363, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_3], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_364, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf232, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf233 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf232, primals_365, primals_366, primals_367, primals_368, buf233, 49152, grid=grid(49152), stream=stream0)
        buf234 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf235 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_78, mul_55, features_29_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf235, buf233, 3072, 16, grid=grid(3072), stream=stream0)
        buf236 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_370, reinterpret_tensor(buf235, (4, 768), (768, 1), 0), reinterpret_tensor(primals_369, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf236)
        del primals_370
        buf237 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_80, mul_56], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf236, buf237, 192, grid=grid(192), stream=stream0)
        buf238 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_372, buf237, reinterpret_tensor(primals_371, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf238)
        del primals_372
        buf239 = buf233; del buf233  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_78, mul_55, mul_57], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf239, buf238, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_29_conv_7], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_373, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf241 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_29_conv_8, add_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf228, buf240, primals_374, primals_375, primals_376, primals_377, buf241, 12288, grid=grid(12288), stream=stream0)
        del primals_377
        # Topologically Sorted Source Nodes: [features_30_conv_0], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_378, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf243 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf244 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [features_30_conv_1, sigmoid_83, mul_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf244, buf242, primals_379, primals_380, primals_381, primals_382, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_3], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_383, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf245, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf246 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf245, primals_384, primals_385, primals_386, primals_387, buf246, 49152, grid=grid(49152), stream=stream0)
        buf247 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf248 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_85, mul_59, features_30_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf248, buf246, 3072, 16, grid=grid(3072), stream=stream0)
        buf249 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_389, reinterpret_tensor(buf248, (4, 768), (768, 1), 0), reinterpret_tensor(primals_388, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf249)
        del primals_389
        buf250 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_87, mul_60], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf249, buf250, 192, grid=grid(192), stream=stream0)
        buf251 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_391, buf250, reinterpret_tensor(primals_390, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf251)
        del primals_391
        buf252 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_85, mul_59, mul_61], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf252, buf251, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_30_conv_7], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf254 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_30_conv_8, add_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf241, buf253, primals_393, primals_394, primals_395, primals_396, buf254, 12288, grid=grid(12288), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [features_31_conv_0], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_397, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf256 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf257 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [features_31_conv_1, sigmoid_90, mul_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf257, buf255, primals_398, primals_399, primals_400, primals_401, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_3], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_402, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf258, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf259 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf258, primals_403, primals_404, primals_405, primals_406, buf259, 49152, grid=grid(49152), stream=stream0)
        buf260 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf261 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_92, mul_63, features_31_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf261, buf259, 3072, 16, grid=grid(3072), stream=stream0)
        buf262 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_408, reinterpret_tensor(buf261, (4, 768), (768, 1), 0), reinterpret_tensor(primals_407, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf262)
        del primals_408
        buf263 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_94, mul_64], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf262, buf263, 192, grid=grid(192), stream=stream0)
        buf264 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_410, buf263, reinterpret_tensor(primals_409, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf264)
        del primals_410
        buf265 = buf259; del buf259  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_92, mul_63, mul_65], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf265, buf264, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_31_conv_7], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_411, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf266, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf267 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_31_conv_8, add_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf254, buf266, primals_412, primals_413, primals_414, primals_415, buf267, 12288, grid=grid(12288), stream=stream0)
        del primals_415
        # Topologically Sorted Source Nodes: [features_32_conv_0], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_416, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf269 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf270 = buf269; del buf269  # reuse
        # Topologically Sorted Source Nodes: [features_32_conv_1, sigmoid_97, mul_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf270, buf268, primals_417, primals_418, primals_419, primals_420, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_3], Original ATen: [aten.convolution]
        buf271 = extern_kernels.convolution(buf270, primals_421, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf271, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf272 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf271, primals_422, primals_423, primals_424, primals_425, buf272, 49152, grid=grid(49152), stream=stream0)
        buf273 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf274 = buf273; del buf273  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_99, mul_67, features_32_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf274, buf272, 3072, 16, grid=grid(3072), stream=stream0)
        buf275 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_427, reinterpret_tensor(buf274, (4, 768), (768, 1), 0), reinterpret_tensor(primals_426, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf275)
        del primals_427
        buf276 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_101, mul_68], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf275, buf276, 192, grid=grid(192), stream=stream0)
        buf277 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_429, buf276, reinterpret_tensor(primals_428, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf277)
        del primals_429
        buf278 = buf272; del buf272  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_99, mul_67, mul_69], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf278, buf277, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_32_conv_7], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_430, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf280 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_32_conv_8, add_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf267, buf279, primals_431, primals_432, primals_433, primals_434, buf280, 12288, grid=grid(12288), stream=stream0)
        del primals_434
        # Topologically Sorted Source Nodes: [features_33_conv_0], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_435, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf282 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf283 = buf282; del buf282  # reuse
        # Topologically Sorted Source Nodes: [features_33_conv_1, sigmoid_104, mul_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf283, buf281, primals_436, primals_437, primals_438, primals_439, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_3], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_440, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf284, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf285 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf284, primals_441, primals_442, primals_443, primals_444, buf285, 49152, grid=grid(49152), stream=stream0)
        buf286 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf287 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_106, mul_71, features_33_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf287, buf285, 3072, 16, grid=grid(3072), stream=stream0)
        buf288 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_446, reinterpret_tensor(buf287, (4, 768), (768, 1), 0), reinterpret_tensor(primals_445, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf288)
        del primals_446
        buf289 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_108, mul_72], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf288, buf289, 192, grid=grid(192), stream=stream0)
        buf290 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_448, buf289, reinterpret_tensor(primals_447, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf290)
        del primals_448
        buf291 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_106, mul_71, mul_73], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf291, buf290, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_33_conv_7], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_449, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf293 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_33_conv_8, add_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf280, buf292, primals_450, primals_451, primals_452, primals_453, buf293, 12288, grid=grid(12288), stream=stream0)
        del primals_453
        # Topologically Sorted Source Nodes: [features_34_conv_0], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_454, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf295 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf296 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [features_34_conv_1, sigmoid_111, mul_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf296, buf294, primals_455, primals_456, primals_457, primals_458, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_3], Original ATen: [aten.convolution]
        buf297 = extern_kernels.convolution(buf296, primals_459, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf297, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf298 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf297, primals_460, primals_461, primals_462, primals_463, buf298, 49152, grid=grid(49152), stream=stream0)
        buf299 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf300 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_113, mul_75, features_34_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf300, buf298, 3072, 16, grid=grid(3072), stream=stream0)
        buf301 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_465, reinterpret_tensor(buf300, (4, 768), (768, 1), 0), reinterpret_tensor(primals_464, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf301)
        del primals_465
        buf302 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_115, mul_76], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf301, buf302, 192, grid=grid(192), stream=stream0)
        buf303 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_467, buf302, reinterpret_tensor(primals_466, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf303)
        del primals_467
        buf304 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_113, mul_75, mul_77], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf304, buf303, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_34_conv_7], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf306 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_34_conv_8, add_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf293, buf305, primals_469, primals_470, primals_471, primals_472, buf306, 12288, grid=grid(12288), stream=stream0)
        del primals_472
        # Topologically Sorted Source Nodes: [features_35_conv_0], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_473, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf308 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf309 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [features_35_conv_1, sigmoid_118, mul_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf309, buf307, primals_474, primals_475, primals_476, primals_477, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_3], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_478, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf310, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf311 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf310, primals_479, primals_480, primals_481, primals_482, buf311, 49152, grid=grid(49152), stream=stream0)
        buf312 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf313 = buf312; del buf312  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_120, mul_79, features_35_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf313, buf311, 3072, 16, grid=grid(3072), stream=stream0)
        buf314 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_484, reinterpret_tensor(buf313, (4, 768), (768, 1), 0), reinterpret_tensor(primals_483, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf314)
        del primals_484
        buf315 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_122, mul_80], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf314, buf315, 192, grid=grid(192), stream=stream0)
        buf316 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_486, buf315, reinterpret_tensor(primals_485, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf316)
        del primals_486
        buf317 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_120, mul_79, mul_81], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf317, buf316, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_35_conv_7], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, primals_487, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf319 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_35_conv_8, add_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf306, buf318, primals_488, primals_489, primals_490, primals_491, buf319, 12288, grid=grid(12288), stream=stream0)
        del primals_491
        # Topologically Sorted Source Nodes: [features_36_conv_0], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf321 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        buf322 = buf321; del buf321  # reuse
        # Topologically Sorted Source Nodes: [features_36_conv_1, sigmoid_125, mul_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_24.run(buf322, buf320, primals_493, primals_494, primals_495, primals_496, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_3], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_497, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf323, (4, 768, 4, 4), (12288, 1, 3072, 768))
        buf324 = empty_strided_cuda((4, 768, 4, 4), (12288, 1, 3072, 768), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf323, primals_498, primals_499, primals_500, primals_501, buf324, 49152, grid=grid(49152), stream=stream0)
        buf325 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 3072, 3072), torch.float32)
        buf326 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_127, mul_83, features_36_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_26.run(buf326, buf324, 3072, 16, grid=grid(3072), stream=stream0)
        buf327 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_503, reinterpret_tensor(buf326, (4, 768), (768, 1), 0), reinterpret_tensor(primals_502, (768, 48), (1, 768), 0), alpha=1, beta=1, out=buf327)
        del primals_503
        buf328 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_129, mul_84], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf327, buf328, 192, grid=grid(192), stream=stream0)
        buf329 = empty_strided_cuda((4, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_505, buf328, reinterpret_tensor(primals_504, (48, 768), (1, 48), 0), alpha=1, beta=1, out=buf329)
        del primals_505
        buf330 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_127, mul_83, mul_85], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_28.run(buf330, buf329, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_36_conv_7], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_506, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 192, 4, 4), (3072, 1, 768, 192))
        buf332 = empty_strided_cuda((4, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [features_36_conv_8, add_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_29.run(buf319, buf331, primals_507, primals_508, primals_509, primals_510, buf332, 12288, grid=grid(12288), stream=stream0)
        del primals_510
        # Topologically Sorted Source Nodes: [features_37_conv_0], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_511, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (4, 1152, 4, 4), (18432, 1, 4608, 1152))
        buf334 = empty_strided_cuda((4, 1152, 4, 4), (18432, 1, 4608, 1152), torch.float32)
        buf335 = buf334; del buf334  # reuse
        # Topologically Sorted Source Nodes: [features_37_conv_1, sigmoid_132, mul_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_30.run(buf335, buf333, primals_512, primals_513, primals_514, primals_515, 73728, grid=grid(73728), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_3], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_516, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf336, (4, 1152, 4, 4), (18432, 1, 4608, 1152))
        buf337 = empty_strided_cuda((4, 1152, 4, 4), (18432, 1, 4608, 1152), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_31.run(buf336, primals_517, primals_518, primals_519, primals_520, buf337, 73728, grid=grid(73728), stream=stream0)
        buf338 = empty_strided_cuda((4, 1152, 1, 1), (1152, 1, 4608, 4608), torch.float32)
        buf339 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_134, mul_87, features_37_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_32.run(buf339, buf337, 4608, 16, grid=grid(4608), stream=stream0)
        buf340 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_522, reinterpret_tensor(buf339, (4, 1152), (1152, 1), 0), reinterpret_tensor(primals_521, (1152, 48), (1, 1152), 0), alpha=1, beta=1, out=buf340)
        del primals_522
        buf341 = empty_strided_cuda((4, 48), (48, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_136, mul_88], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_27.run(buf340, buf341, 192, grid=grid(192), stream=stream0)
        buf342 = empty_strided_cuda((4, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_524, buf341, reinterpret_tensor(primals_523, (48, 1152), (1, 48), 0), alpha=1, beta=1, out=buf342)
        del primals_524
        buf343 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_134, mul_87, mul_89], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_33.run(buf343, buf342, 73728, grid=grid(73728), stream=stream0)
        # Topologically Sorted Source Nodes: [features_37_conv_7], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_525, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf345 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_37_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_34.run(buf344, primals_526, primals_527, primals_528, primals_529, buf345, 16384, grid=grid(16384), stream=stream0)
        del primals_529
        # Topologically Sorted Source Nodes: [features_38_conv_0], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_530, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf347 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf348 = buf347; del buf347  # reuse
        # Topologically Sorted Source Nodes: [features_38_conv_1, sigmoid_139, mul_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf348, buf346, primals_531, primals_532, primals_533, primals_534, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_3], Original ATen: [aten.convolution]
        buf349 = extern_kernels.convolution(buf348, primals_535, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf349, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf350 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf349, primals_536, primals_537, primals_538, primals_539, buf350, 98304, grid=grid(98304), stream=stream0)
        buf351 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf352 = buf351; del buf351  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_141, mul_91, features_38_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf352, buf350, 6144, 16, grid=grid(6144), stream=stream0)
        buf353 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_541, reinterpret_tensor(buf352, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_540, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf353)
        del primals_541
        buf354 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_143, mul_92], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf353, buf354, 256, grid=grid(256), stream=stream0)
        buf355 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_543, buf354, reinterpret_tensor(primals_542, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf355)
        del primals_543
        buf356 = buf350; del buf350  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_141, mul_91, mul_93], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf356, buf355, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_38_conv_7], Original ATen: [aten.convolution]
        buf357 = extern_kernels.convolution(buf356, primals_544, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf357, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf358 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_38_conv_8, add_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf345, buf357, primals_545, primals_546, primals_547, primals_548, buf358, 16384, grid=grid(16384), stream=stream0)
        del primals_548
        # Topologically Sorted Source Nodes: [features_39_conv_0], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_549, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf360 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf361 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [features_39_conv_1, sigmoid_146, mul_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf361, buf359, primals_550, primals_551, primals_552, primals_553, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_3], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_554, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf362, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf363 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf362, primals_555, primals_556, primals_557, primals_558, buf363, 98304, grid=grid(98304), stream=stream0)
        buf364 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf365 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_148, mul_95, features_39_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf365, buf363, 6144, 16, grid=grid(6144), stream=stream0)
        buf366 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_560, reinterpret_tensor(buf365, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_559, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf366)
        del primals_560
        buf367 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_150, mul_96], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf366, buf367, 256, grid=grid(256), stream=stream0)
        buf368 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_562, buf367, reinterpret_tensor(primals_561, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf368)
        del primals_562
        buf369 = buf363; del buf363  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_148, mul_95, mul_97], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf369, buf368, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_39_conv_7], Original ATen: [aten.convolution]
        buf370 = extern_kernels.convolution(buf369, primals_563, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf370, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf371 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_39_conv_8, add_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf358, buf370, primals_564, primals_565, primals_566, primals_567, buf371, 16384, grid=grid(16384), stream=stream0)
        del primals_567
        # Topologically Sorted Source Nodes: [features_40_conv_0], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_568, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf373 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf374 = buf373; del buf373  # reuse
        # Topologically Sorted Source Nodes: [features_40_conv_1, sigmoid_153, mul_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf374, buf372, primals_569, primals_570, primals_571, primals_572, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_3], Original ATen: [aten.convolution]
        buf375 = extern_kernels.convolution(buf374, primals_573, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf375, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf376 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf375, primals_574, primals_575, primals_576, primals_577, buf376, 98304, grid=grid(98304), stream=stream0)
        buf377 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf378 = buf377; del buf377  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_155, mul_99, features_40_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf378, buf376, 6144, 16, grid=grid(6144), stream=stream0)
        buf379 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_579, reinterpret_tensor(buf378, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_578, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf379)
        del primals_579
        buf380 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_157, mul_100], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf379, buf380, 256, grid=grid(256), stream=stream0)
        buf381 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_581, buf380, reinterpret_tensor(primals_580, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf381)
        del primals_581
        buf382 = buf376; del buf376  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_155, mul_99, mul_101], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf382, buf381, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_40_conv_7], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_582, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf384 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_40_conv_8, add_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf371, buf383, primals_583, primals_584, primals_585, primals_586, buf384, 16384, grid=grid(16384), stream=stream0)
        del primals_586
        # Topologically Sorted Source Nodes: [features_41_conv_0], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_587, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf386 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf387 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [features_41_conv_1, sigmoid_160, mul_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf387, buf385, primals_588, primals_589, primals_590, primals_591, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_41_conv_3], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf387, primals_592, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf388, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf389 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf388, primals_593, primals_594, primals_595, primals_596, buf389, 98304, grid=grid(98304), stream=stream0)
        buf390 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf391 = buf390; del buf390  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_162, mul_103, features_41_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf391, buf389, 6144, 16, grid=grid(6144), stream=stream0)
        buf392 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_598, reinterpret_tensor(buf391, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_597, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf392)
        del primals_598
        buf393 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_164, mul_104], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf392, buf393, 256, grid=grid(256), stream=stream0)
        buf394 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_600, buf393, reinterpret_tensor(primals_599, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf394)
        del primals_600
        buf395 = buf389; del buf389  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_162, mul_103, mul_105], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf395, buf394, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_41_conv_7], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_601, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf397 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_41_conv_8, add_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf384, buf396, primals_602, primals_603, primals_604, primals_605, buf397, 16384, grid=grid(16384), stream=stream0)
        del primals_605
        # Topologically Sorted Source Nodes: [features_42_conv_0], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_606, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf399 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf400 = buf399; del buf399  # reuse
        # Topologically Sorted Source Nodes: [features_42_conv_1, sigmoid_167, mul_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf400, buf398, primals_607, primals_608, primals_609, primals_610, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_42_conv_3], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_611, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf401, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf402 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf401, primals_612, primals_613, primals_614, primals_615, buf402, 98304, grid=grid(98304), stream=stream0)
        buf403 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf404 = buf403; del buf403  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_169, mul_107, features_42_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf404, buf402, 6144, 16, grid=grid(6144), stream=stream0)
        buf405 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_617, reinterpret_tensor(buf404, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_616, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf405)
        del primals_617
        buf406 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_171, mul_108], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf405, buf406, 256, grid=grid(256), stream=stream0)
        buf407 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_619, buf406, reinterpret_tensor(primals_618, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf407)
        del primals_619
        buf408 = buf402; del buf402  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_169, mul_107, mul_109], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf408, buf407, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_42_conv_7], Original ATen: [aten.convolution]
        buf409 = extern_kernels.convolution(buf408, primals_620, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf409, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf410 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_42_conv_8, add_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf397, buf409, primals_621, primals_622, primals_623, primals_624, buf410, 16384, grid=grid(16384), stream=stream0)
        del primals_624
        # Topologically Sorted Source Nodes: [features_43_conv_0], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_625, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf412 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf413 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [features_43_conv_1, sigmoid_174, mul_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf413, buf411, primals_626, primals_627, primals_628, primals_629, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_43_conv_3], Original ATen: [aten.convolution]
        buf414 = extern_kernels.convolution(buf413, primals_630, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf414, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf415 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf414, primals_631, primals_632, primals_633, primals_634, buf415, 98304, grid=grid(98304), stream=stream0)
        buf416 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf417 = buf416; del buf416  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_176, mul_111, features_43_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf417, buf415, 6144, 16, grid=grid(6144), stream=stream0)
        buf418 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_636, reinterpret_tensor(buf417, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_635, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf418)
        del primals_636
        buf419 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_178, mul_112], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf418, buf419, 256, grid=grid(256), stream=stream0)
        buf420 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_638, buf419, reinterpret_tensor(primals_637, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf420)
        del primals_638
        buf421 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_176, mul_111, mul_113], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf421, buf420, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_43_conv_7], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_639, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf423 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_43_conv_8, add_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf410, buf422, primals_640, primals_641, primals_642, primals_643, buf423, 16384, grid=grid(16384), stream=stream0)
        del primals_643
        # Topologically Sorted Source Nodes: [features_44_conv_0], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_644, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf425 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf426 = buf425; del buf425  # reuse
        # Topologically Sorted Source Nodes: [features_44_conv_1, sigmoid_181, mul_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf426, buf424, primals_645, primals_646, primals_647, primals_648, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_44_conv_3], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_649, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf427, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf428 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf427, primals_650, primals_651, primals_652, primals_653, buf428, 98304, grid=grid(98304), stream=stream0)
        buf429 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf430 = buf429; del buf429  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_183, mul_115, features_44_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf430, buf428, 6144, 16, grid=grid(6144), stream=stream0)
        buf431 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_655, reinterpret_tensor(buf430, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_654, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf431)
        del primals_655
        buf432 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_185, mul_116], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf431, buf432, 256, grid=grid(256), stream=stream0)
        buf433 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_657, buf432, reinterpret_tensor(primals_656, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf433)
        del primals_657
        buf434 = buf428; del buf428  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_183, mul_115, mul_117], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf434, buf433, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_44_conv_7], Original ATen: [aten.convolution]
        buf435 = extern_kernels.convolution(buf434, primals_658, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf435, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf436 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_44_conv_8, add_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf423, buf435, primals_659, primals_660, primals_661, primals_662, buf436, 16384, grid=grid(16384), stream=stream0)
        del primals_662
        # Topologically Sorted Source Nodes: [features_45_conv_0], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, primals_663, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf438 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf439 = buf438; del buf438  # reuse
        # Topologically Sorted Source Nodes: [features_45_conv_1, sigmoid_188, mul_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf439, buf437, primals_664, primals_665, primals_666, primals_667, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_45_conv_3], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, primals_668, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf440, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf441 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf440, primals_669, primals_670, primals_671, primals_672, buf441, 98304, grid=grid(98304), stream=stream0)
        buf442 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf443 = buf442; del buf442  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_190, mul_119, features_45_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf443, buf441, 6144, 16, grid=grid(6144), stream=stream0)
        buf444 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_674, reinterpret_tensor(buf443, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_673, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf444)
        del primals_674
        buf445 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_192, mul_120], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf444, buf445, 256, grid=grid(256), stream=stream0)
        buf446 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_676, buf445, reinterpret_tensor(primals_675, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf446)
        del primals_676
        buf447 = buf441; del buf441  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_190, mul_119, mul_121], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf447, buf446, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_45_conv_7], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf447, primals_677, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf449 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_45_conv_8, add_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf436, buf448, primals_678, primals_679, primals_680, primals_681, buf449, 16384, grid=grid(16384), stream=stream0)
        del primals_681
        # Topologically Sorted Source Nodes: [features_46_conv_0], Original ATen: [aten.convolution]
        buf450 = extern_kernels.convolution(buf449, primals_682, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf450, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf451 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf452 = buf451; del buf451  # reuse
        # Topologically Sorted Source Nodes: [features_46_conv_1, sigmoid_195, mul_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf452, buf450, primals_683, primals_684, primals_685, primals_686, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_46_conv_3], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_687, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf453, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf454 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf453, primals_688, primals_689, primals_690, primals_691, buf454, 98304, grid=grid(98304), stream=stream0)
        buf455 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf456 = buf455; del buf455  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_197, mul_123, features_46_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf456, buf454, 6144, 16, grid=grid(6144), stream=stream0)
        buf457 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_693, reinterpret_tensor(buf456, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_692, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf457)
        del primals_693
        buf458 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_199, mul_124], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf457, buf458, 256, grid=grid(256), stream=stream0)
        buf459 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_695, buf458, reinterpret_tensor(primals_694, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf459)
        del primals_695
        buf460 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_197, mul_123, mul_125], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf460, buf459, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_46_conv_7], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_696, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf462 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_46_conv_8, add_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf449, buf461, primals_697, primals_698, primals_699, primals_700, buf462, 16384, grid=grid(16384), stream=stream0)
        del primals_700
        # Topologically Sorted Source Nodes: [features_47_conv_0], Original ATen: [aten.convolution]
        buf463 = extern_kernels.convolution(buf462, primals_701, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf464 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf465 = buf464; del buf464  # reuse
        # Topologically Sorted Source Nodes: [features_47_conv_1, sigmoid_202, mul_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf465, buf463, primals_702, primals_703, primals_704, primals_705, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_47_conv_3], Original ATen: [aten.convolution]
        buf466 = extern_kernels.convolution(buf465, primals_706, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf466, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf467 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf466, primals_707, primals_708, primals_709, primals_710, buf467, 98304, grid=grid(98304), stream=stream0)
        buf468 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf469 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_204, mul_127, features_47_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf469, buf467, 6144, 16, grid=grid(6144), stream=stream0)
        buf470 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_712, reinterpret_tensor(buf469, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_711, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf470)
        del primals_712
        buf471 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_206, mul_128], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf470, buf471, 256, grid=grid(256), stream=stream0)
        buf472 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_714, buf471, reinterpret_tensor(primals_713, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf472)
        del primals_714
        buf473 = buf467; del buf467  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_204, mul_127, mul_129], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf473, buf472, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_47_conv_7], Original ATen: [aten.convolution]
        buf474 = extern_kernels.convolution(buf473, primals_715, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf474, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf475 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_47_conv_8, add_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf462, buf474, primals_716, primals_717, primals_718, primals_719, buf475, 16384, grid=grid(16384), stream=stream0)
        del primals_719
        # Topologically Sorted Source Nodes: [features_48_conv_0], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_720, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf477 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf478 = buf477; del buf477  # reuse
        # Topologically Sorted Source Nodes: [features_48_conv_1, sigmoid_209, mul_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf478, buf476, primals_721, primals_722, primals_723, primals_724, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_48_conv_3], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_725, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf479, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf480 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf479, primals_726, primals_727, primals_728, primals_729, buf480, 98304, grid=grid(98304), stream=stream0)
        buf481 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf482 = buf481; del buf481  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_211, mul_131, features_48_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf482, buf480, 6144, 16, grid=grid(6144), stream=stream0)
        buf483 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_731, reinterpret_tensor(buf482, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_730, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf483)
        del primals_731
        buf484 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_213, mul_132], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf483, buf484, 256, grid=grid(256), stream=stream0)
        buf485 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_733, buf484, reinterpret_tensor(primals_732, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf485)
        del primals_733
        buf486 = buf480; del buf480  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_211, mul_131, mul_133], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf486, buf485, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_48_conv_7], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, primals_734, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf488 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_48_conv_8, add_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf475, buf487, primals_735, primals_736, primals_737, primals_738, buf488, 16384, grid=grid(16384), stream=stream0)
        del primals_738
        # Topologically Sorted Source Nodes: [features_49_conv_0], Original ATen: [aten.convolution]
        buf489 = extern_kernels.convolution(buf488, primals_739, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf489, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf490 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf491 = buf490; del buf490  # reuse
        # Topologically Sorted Source Nodes: [features_49_conv_1, sigmoid_216, mul_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf491, buf489, primals_740, primals_741, primals_742, primals_743, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_49_conv_3], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf491, primals_744, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf492, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf493 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf492, primals_745, primals_746, primals_747, primals_748, buf493, 98304, grid=grid(98304), stream=stream0)
        buf494 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf495 = buf494; del buf494  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_218, mul_135, features_49_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf495, buf493, 6144, 16, grid=grid(6144), stream=stream0)
        buf496 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_750, reinterpret_tensor(buf495, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_749, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf496)
        del primals_750
        buf497 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_220, mul_136], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf496, buf497, 256, grid=grid(256), stream=stream0)
        buf498 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_752, buf497, reinterpret_tensor(primals_751, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf498)
        del primals_752
        buf499 = buf493; del buf493  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_218, mul_135, mul_137], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf499, buf498, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_49_conv_7], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf499, primals_753, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf500, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf501 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_49_conv_8, add_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf488, buf500, primals_754, primals_755, primals_756, primals_757, buf501, 16384, grid=grid(16384), stream=stream0)
        del primals_757
        # Topologically Sorted Source Nodes: [features_50_conv_0], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_758, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf502, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf503 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf504 = buf503; del buf503  # reuse
        # Topologically Sorted Source Nodes: [features_50_conv_1, sigmoid_223, mul_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf504, buf502, primals_759, primals_760, primals_761, primals_762, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_50_conv_3], Original ATen: [aten.convolution]
        buf505 = extern_kernels.convolution(buf504, primals_763, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf505, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf506 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf505, primals_764, primals_765, primals_766, primals_767, buf506, 98304, grid=grid(98304), stream=stream0)
        buf507 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf508 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_225, mul_139, features_50_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf508, buf506, 6144, 16, grid=grid(6144), stream=stream0)
        buf509 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_769, reinterpret_tensor(buf508, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_768, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf509)
        del primals_769
        buf510 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_227, mul_140], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf509, buf510, 256, grid=grid(256), stream=stream0)
        buf511 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_771, buf510, reinterpret_tensor(primals_770, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf511)
        del primals_771
        buf512 = buf506; del buf506  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_225, mul_139, mul_141], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf512, buf511, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_50_conv_7], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(buf512, primals_772, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf513, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf514 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_50_conv_8, add_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf501, buf513, primals_773, primals_774, primals_775, primals_776, buf514, 16384, grid=grid(16384), stream=stream0)
        del primals_776
        # Topologically Sorted Source Nodes: [features_51_conv_0], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_777, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf516 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf517 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [features_51_conv_1, sigmoid_230, mul_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf517, buf515, primals_778, primals_779, primals_780, primals_781, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_51_conv_3], Original ATen: [aten.convolution]
        buf518 = extern_kernels.convolution(buf517, primals_782, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf518, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf519 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf518, primals_783, primals_784, primals_785, primals_786, buf519, 98304, grid=grid(98304), stream=stream0)
        buf520 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf521 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_232, mul_143, features_51_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf521, buf519, 6144, 16, grid=grid(6144), stream=stream0)
        buf522 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_788, reinterpret_tensor(buf521, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_787, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf522)
        del primals_788
        buf523 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_234, mul_144], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf522, buf523, 256, grid=grid(256), stream=stream0)
        buf524 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_790, buf523, reinterpret_tensor(primals_789, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf524)
        del primals_790
        buf525 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_232, mul_143, mul_145], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf525, buf524, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_51_conv_7], Original ATen: [aten.convolution]
        buf526 = extern_kernels.convolution(buf525, primals_791, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf526, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf527 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_51_conv_8, add_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf514, buf526, primals_792, primals_793, primals_794, primals_795, buf527, 16384, grid=grid(16384), stream=stream0)
        del primals_795
        # Topologically Sorted Source Nodes: [features_52_conv_0], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, primals_796, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf528, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf529 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf530 = buf529; del buf529  # reuse
        # Topologically Sorted Source Nodes: [features_52_conv_1, sigmoid_237, mul_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf530, buf528, primals_797, primals_798, primals_799, primals_800, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_52_conv_3], Original ATen: [aten.convolution]
        buf531 = extern_kernels.convolution(buf530, primals_801, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf531, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf532 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf531, primals_802, primals_803, primals_804, primals_805, buf532, 98304, grid=grid(98304), stream=stream0)
        buf533 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf534 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_239, mul_147, features_52_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf534, buf532, 6144, 16, grid=grid(6144), stream=stream0)
        buf535 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_807, reinterpret_tensor(buf534, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_806, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf535)
        del primals_807
        buf536 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_241, mul_148], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf535, buf536, 256, grid=grid(256), stream=stream0)
        buf537 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_809, buf536, reinterpret_tensor(primals_808, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf537)
        del primals_809
        buf538 = buf532; del buf532  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_239, mul_147, mul_149], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf538, buf537, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_52_conv_7], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_810, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf540 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_52_conv_8, add_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf527, buf539, primals_811, primals_812, primals_813, primals_814, buf540, 16384, grid=grid(16384), stream=stream0)
        del primals_814
        # Topologically Sorted Source Nodes: [features_53_conv_0], Original ATen: [aten.convolution]
        buf541 = extern_kernels.convolution(buf540, primals_815, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf542 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf543 = buf542; del buf542  # reuse
        # Topologically Sorted Source Nodes: [features_53_conv_1, sigmoid_244, mul_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf543, buf541, primals_816, primals_817, primals_818, primals_819, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_53_conv_3], Original ATen: [aten.convolution]
        buf544 = extern_kernels.convolution(buf543, primals_820, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf544, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf545 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf544, primals_821, primals_822, primals_823, primals_824, buf545, 98304, grid=grid(98304), stream=stream0)
        buf546 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf547 = buf546; del buf546  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_246, mul_151, features_53_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf547, buf545, 6144, 16, grid=grid(6144), stream=stream0)
        buf548 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_826, reinterpret_tensor(buf547, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_825, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf548)
        del primals_826
        buf549 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_248, mul_152], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf548, buf549, 256, grid=grid(256), stream=stream0)
        buf550 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_828, buf549, reinterpret_tensor(primals_827, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf550)
        del primals_828
        buf551 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_246, mul_151, mul_153], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf551, buf550, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_53_conv_7], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_829, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf553 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_53_conv_8, add_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf540, buf552, primals_830, primals_831, primals_832, primals_833, buf553, 16384, grid=grid(16384), stream=stream0)
        del primals_833
        # Topologically Sorted Source Nodes: [features_54_conv_0], Original ATen: [aten.convolution]
        buf554 = extern_kernels.convolution(buf553, primals_834, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf554, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf555 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf556 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [features_54_conv_1, sigmoid_251, mul_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf556, buf554, primals_835, primals_836, primals_837, primals_838, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_54_conv_3], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, primals_839, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf557, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf558 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf557, primals_840, primals_841, primals_842, primals_843, buf558, 98304, grid=grid(98304), stream=stream0)
        buf559 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf560 = buf559; del buf559  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_253, mul_155, features_54_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf560, buf558, 6144, 16, grid=grid(6144), stream=stream0)
        buf561 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_845, reinterpret_tensor(buf560, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_844, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf561)
        del primals_845
        buf562 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_255, mul_156], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf561, buf562, 256, grid=grid(256), stream=stream0)
        buf563 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_847, buf562, reinterpret_tensor(primals_846, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf563)
        del primals_847
        buf564 = buf558; del buf558  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_253, mul_155, mul_157], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf564, buf563, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_54_conv_7], Original ATen: [aten.convolution]
        buf565 = extern_kernels.convolution(buf564, primals_848, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf565, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf566 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_54_conv_8, add_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf553, buf565, primals_849, primals_850, primals_851, primals_852, buf566, 16384, grid=grid(16384), stream=stream0)
        del primals_852
        # Topologically Sorted Source Nodes: [features_55_conv_0], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, primals_853, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf568 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf569 = buf568; del buf568  # reuse
        # Topologically Sorted Source Nodes: [features_55_conv_1, sigmoid_258, mul_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf569, buf567, primals_854, primals_855, primals_856, primals_857, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_55_conv_3], Original ATen: [aten.convolution]
        buf570 = extern_kernels.convolution(buf569, primals_858, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf570, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf571 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf570, primals_859, primals_860, primals_861, primals_862, buf571, 98304, grid=grid(98304), stream=stream0)
        buf572 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf573 = buf572; del buf572  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_260, mul_159, features_55_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf573, buf571, 6144, 16, grid=grid(6144), stream=stream0)
        buf574 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_864, reinterpret_tensor(buf573, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_863, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf574)
        del primals_864
        buf575 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_262, mul_160], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf574, buf575, 256, grid=grid(256), stream=stream0)
        buf576 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_866, buf575, reinterpret_tensor(primals_865, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf576)
        del primals_866
        buf577 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_260, mul_159, mul_161], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf577, buf576, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_55_conv_7], Original ATen: [aten.convolution]
        buf578 = extern_kernels.convolution(buf577, primals_867, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf578, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf579 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_55_conv_8, add_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf566, buf578, primals_868, primals_869, primals_870, primals_871, buf579, 16384, grid=grid(16384), stream=stream0)
        del primals_871
        # Topologically Sorted Source Nodes: [features_56_conv_0], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf579, primals_872, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf580, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf581 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf582 = buf581; del buf581  # reuse
        # Topologically Sorted Source Nodes: [features_56_conv_1, sigmoid_265, mul_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf582, buf580, primals_873, primals_874, primals_875, primals_876, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_56_conv_3], Original ATen: [aten.convolution]
        buf583 = extern_kernels.convolution(buf582, primals_877, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf583, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf584 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf583, primals_878, primals_879, primals_880, primals_881, buf584, 98304, grid=grid(98304), stream=stream0)
        buf585 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf586 = buf585; del buf585  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_267, mul_163, features_56_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf586, buf584, 6144, 16, grid=grid(6144), stream=stream0)
        buf587 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_883, reinterpret_tensor(buf586, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_882, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf587)
        del primals_883
        buf588 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_269, mul_164], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf587, buf588, 256, grid=grid(256), stream=stream0)
        buf589 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_885, buf588, reinterpret_tensor(primals_884, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf589)
        del primals_885
        buf590 = buf584; del buf584  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_267, mul_163, mul_165], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf590, buf589, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_56_conv_7], Original ATen: [aten.convolution]
        buf591 = extern_kernels.convolution(buf590, primals_886, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf591, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf592 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_56_conv_8, add_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf579, buf591, primals_887, primals_888, primals_889, primals_890, buf592, 16384, grid=grid(16384), stream=stream0)
        del primals_890
        # Topologically Sorted Source Nodes: [features_57_conv_0], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, primals_891, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf594 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf595 = buf594; del buf594  # reuse
        # Topologically Sorted Source Nodes: [features_57_conv_1, sigmoid_272, mul_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf595, buf593, primals_892, primals_893, primals_894, primals_895, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_57_conv_3], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_896, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf596, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf597 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf596, primals_897, primals_898, primals_899, primals_900, buf597, 98304, grid=grid(98304), stream=stream0)
        buf598 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf599 = buf598; del buf598  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_274, mul_167, features_57_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf599, buf597, 6144, 16, grid=grid(6144), stream=stream0)
        buf600 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_902, reinterpret_tensor(buf599, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_901, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf600)
        del primals_902
        buf601 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_276, mul_168], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf600, buf601, 256, grid=grid(256), stream=stream0)
        buf602 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_904, buf601, reinterpret_tensor(primals_903, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf602)
        del primals_904
        buf603 = buf597; del buf597  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_274, mul_167, mul_169], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf603, buf602, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_57_conv_7], Original ATen: [aten.convolution]
        buf604 = extern_kernels.convolution(buf603, primals_905, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf604, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf605 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_57_conv_8, add_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf592, buf604, primals_906, primals_907, primals_908, primals_909, buf605, 16384, grid=grid(16384), stream=stream0)
        del primals_909
        # Topologically Sorted Source Nodes: [features_58_conv_0], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_910, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf607 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf608 = buf607; del buf607  # reuse
        # Topologically Sorted Source Nodes: [features_58_conv_1, sigmoid_279, mul_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf608, buf606, primals_911, primals_912, primals_913, primals_914, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_58_conv_3], Original ATen: [aten.convolution]
        buf609 = extern_kernels.convolution(buf608, primals_915, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf609, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf610 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf609, primals_916, primals_917, primals_918, primals_919, buf610, 98304, grid=grid(98304), stream=stream0)
        buf611 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf612 = buf611; del buf611  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_281, mul_171, features_58_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf612, buf610, 6144, 16, grid=grid(6144), stream=stream0)
        buf613 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_921, reinterpret_tensor(buf612, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_920, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf613)
        del primals_921
        buf614 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_283, mul_172], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf613, buf614, 256, grid=grid(256), stream=stream0)
        buf615 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_923, buf614, reinterpret_tensor(primals_922, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf615)
        del primals_923
        buf616 = buf610; del buf610  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_281, mul_171, mul_173], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf616, buf615, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_58_conv_7], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, primals_924, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf617, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf618 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_58_conv_8, add_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf605, buf617, primals_925, primals_926, primals_927, primals_928, buf618, 16384, grid=grid(16384), stream=stream0)
        del primals_928
        # Topologically Sorted Source Nodes: [features_59_conv_0], Original ATen: [aten.convolution]
        buf619 = extern_kernels.convolution(buf618, primals_929, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf619, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf620 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf621 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [features_59_conv_1, sigmoid_286, mul_174], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf621, buf619, primals_930, primals_931, primals_932, primals_933, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_59_conv_3], Original ATen: [aten.convolution]
        buf622 = extern_kernels.convolution(buf621, primals_934, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf622, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf623 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf622, primals_935, primals_936, primals_937, primals_938, buf623, 98304, grid=grid(98304), stream=stream0)
        buf624 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf625 = buf624; del buf624  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_288, mul_175, features_59_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf625, buf623, 6144, 16, grid=grid(6144), stream=stream0)
        buf626 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_940, reinterpret_tensor(buf625, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_939, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf626)
        del primals_940
        buf627 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_290, mul_176], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf626, buf627, 256, grid=grid(256), stream=stream0)
        buf628 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_942, buf627, reinterpret_tensor(primals_941, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf628)
        del primals_942
        buf629 = buf623; del buf623  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_288, mul_175, mul_177], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf629, buf628, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_59_conv_7], Original ATen: [aten.convolution]
        buf630 = extern_kernels.convolution(buf629, primals_943, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf631 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_59_conv_8, add_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf618, buf630, primals_944, primals_945, primals_946, primals_947, buf631, 16384, grid=grid(16384), stream=stream0)
        del primals_947
        # Topologically Sorted Source Nodes: [features_60_conv_0], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_948, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf633 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf634 = buf633; del buf633  # reuse
        # Topologically Sorted Source Nodes: [features_60_conv_1, sigmoid_293, mul_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf634, buf632, primals_949, primals_950, primals_951, primals_952, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_60_conv_3], Original ATen: [aten.convolution]
        buf635 = extern_kernels.convolution(buf634, primals_953, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf635, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf636 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_36.run(buf635, primals_954, primals_955, primals_956, primals_957, buf636, 98304, grid=grid(98304), stream=stream0)
        buf637 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        buf638 = buf637; del buf637  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_295, mul_179, features_60_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_mul_sigmoid_37.run(buf638, buf636, 6144, 16, grid=grid(6144), stream=stream0)
        buf639 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_959, reinterpret_tensor(buf638, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_958, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf639)
        del primals_959
        buf640 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_297, mul_180], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf639, buf640, 256, grid=grid(256), stream=stream0)
        buf641 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_961, buf640, reinterpret_tensor(primals_960, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf641)
        del primals_961
        buf642 = buf636; del buf636  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_295, mul_179, mul_181], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_39.run(buf642, buf641, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_60_conv_7], Original ATen: [aten.convolution]
        buf643 = extern_kernels.convolution(buf642, primals_962, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf643, (4, 256, 4, 4), (4096, 1, 1024, 256))
        buf644 = empty_strided_cuda((4, 256, 4, 4), (4096, 1, 1024, 256), torch.float32)
        # Topologically Sorted Source Nodes: [features_60_conv_8, add_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_40.run(buf631, buf643, primals_963, primals_964, primals_965, primals_966, buf644, 16384, grid=grid(16384), stream=stream0)
        del primals_966
        # Topologically Sorted Source Nodes: [features_61_conv_0], Original ATen: [aten.convolution]
        buf645 = extern_kernels.convolution(buf644, primals_967, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (4, 1536, 4, 4), (24576, 1, 6144, 1536))
        buf646 = empty_strided_cuda((4, 1536, 4, 4), (24576, 1, 6144, 1536), torch.float32)
        buf647 = buf646; del buf646  # reuse
        # Topologically Sorted Source Nodes: [features_61_conv_1, sigmoid_300, mul_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_35.run(buf647, buf645, primals_968, primals_969, primals_970, primals_971, 98304, grid=grid(98304), stream=stream0)
        # Topologically Sorted Source Nodes: [features_61_conv_3], Original ATen: [aten.convolution]
        buf648 = extern_kernels.convolution(buf647, primals_972, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf648, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf649 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_41.run(buf648, primals_973, primals_974, primals_975, primals_976, buf649, 24576, grid=grid(24576), stream=stream0)
        buf650 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 6144, 6144), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_302, mul_183, features_61_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_42.run(buf649, buf650, 6144, grid=grid(6144), stream=stream0)
        buf651 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_978, reinterpret_tensor(buf650, (4, 1536), (1536, 1), 0), reinterpret_tensor(primals_977, (1536, 64), (1, 1536), 0), alpha=1, beta=1, out=buf651)
        del primals_978
        buf652 = empty_strided_cuda((4, 64), (64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_304, mul_184], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_38.run(buf651, buf652, 256, grid=grid(256), stream=stream0)
        buf653 = empty_strided_cuda((4, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_980, buf652, reinterpret_tensor(primals_979, (64, 1536), (1, 64), 0), alpha=1, beta=1, out=buf653)
        del primals_980
        buf654 = buf649; del buf649  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_302, mul_183, mul_185], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_43.run(buf654, buf653, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [features_61_conv_7], Original ATen: [aten.convolution]
        buf655 = extern_kernels.convolution(buf654, primals_981, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf655, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf656 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_61_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_44.run(buf655, primals_982, primals_983, primals_984, primals_985, buf656, 8192, grid=grid(8192), stream=stream0)
        del primals_985
        # Topologically Sorted Source Nodes: [features_62_conv_0], Original ATen: [aten.convolution]
        buf657 = extern_kernels.convolution(buf656, primals_986, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf658 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf659 = buf658; del buf658  # reuse
        # Topologically Sorted Source Nodes: [features_62_conv_1, sigmoid_307, mul_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf659, buf657, primals_987, primals_988, primals_989, primals_990, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_62_conv_3], Original ATen: [aten.convolution]
        buf660 = extern_kernels.convolution(buf659, primals_991, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf660, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf661 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf660, primals_992, primals_993, primals_994, primals_995, buf661, 49152, grid=grid(49152), stream=stream0)
        buf662 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_309, mul_187, features_62_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf661, buf662, 12288, grid=grid(12288), stream=stream0)
        buf663 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_997, reinterpret_tensor(buf662, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_996, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf663)
        del primals_997
        buf664 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_311, mul_188], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf663, buf664, 512, grid=grid(512), stream=stream0)
        buf665 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_999, buf664, reinterpret_tensor(primals_998, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf665)
        del primals_999
        buf666 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_309, mul_187, mul_189], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf666, buf665, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_62_conv_7], Original ATen: [aten.convolution]
        buf667 = extern_kernels.convolution(buf666, primals_1000, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf667, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf668 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_62_conv_8, add_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf656, buf667, primals_1001, primals_1002, primals_1003, primals_1004, buf668, 8192, grid=grid(8192), stream=stream0)
        del primals_1004
        # Topologically Sorted Source Nodes: [features_63_conv_0], Original ATen: [aten.convolution]
        buf669 = extern_kernels.convolution(buf668, primals_1005, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf669, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf670 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf671 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [features_63_conv_1, sigmoid_314, mul_190], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf671, buf669, primals_1006, primals_1007, primals_1008, primals_1009, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_63_conv_3], Original ATen: [aten.convolution]
        buf672 = extern_kernels.convolution(buf671, primals_1010, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf672, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf673 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf672, primals_1011, primals_1012, primals_1013, primals_1014, buf673, 49152, grid=grid(49152), stream=stream0)
        buf674 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_316, mul_191, features_63_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf673, buf674, 12288, grid=grid(12288), stream=stream0)
        buf675 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1016, reinterpret_tensor(buf674, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1015, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf675)
        del primals_1016
        buf676 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_318, mul_192], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf675, buf676, 512, grid=grid(512), stream=stream0)
        buf677 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1018, buf676, reinterpret_tensor(primals_1017, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf677)
        del primals_1018
        buf678 = buf673; del buf673  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_316, mul_191, mul_193], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf678, buf677, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_63_conv_7], Original ATen: [aten.convolution]
        buf679 = extern_kernels.convolution(buf678, primals_1019, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf679, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf680 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_63_conv_8, add_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf668, buf679, primals_1020, primals_1021, primals_1022, primals_1023, buf680, 8192, grid=grid(8192), stream=stream0)
        del primals_1023
        # Topologically Sorted Source Nodes: [features_64_conv_0], Original ATen: [aten.convolution]
        buf681 = extern_kernels.convolution(buf680, primals_1024, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf681, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf682 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf683 = buf682; del buf682  # reuse
        # Topologically Sorted Source Nodes: [features_64_conv_1, sigmoid_321, mul_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf683, buf681, primals_1025, primals_1026, primals_1027, primals_1028, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_64_conv_3], Original ATen: [aten.convolution]
        buf684 = extern_kernels.convolution(buf683, primals_1029, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf684, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf685 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf684, primals_1030, primals_1031, primals_1032, primals_1033, buf685, 49152, grid=grid(49152), stream=stream0)
        buf686 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_323, mul_195, features_64_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf685, buf686, 12288, grid=grid(12288), stream=stream0)
        buf687 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1035, reinterpret_tensor(buf686, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1034, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf687)
        del primals_1035
        buf688 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_325, mul_196], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf687, buf688, 512, grid=grid(512), stream=stream0)
        buf689 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1037, buf688, reinterpret_tensor(primals_1036, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf689)
        del primals_1037
        buf690 = buf685; del buf685  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_323, mul_195, mul_197], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf690, buf689, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_64_conv_7], Original ATen: [aten.convolution]
        buf691 = extern_kernels.convolution(buf690, primals_1038, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf692 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_64_conv_8, add_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf680, buf691, primals_1039, primals_1040, primals_1041, primals_1042, buf692, 8192, grid=grid(8192), stream=stream0)
        del primals_1042
        # Topologically Sorted Source Nodes: [features_65_conv_0], Original ATen: [aten.convolution]
        buf693 = extern_kernels.convolution(buf692, primals_1043, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf694 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf695 = buf694; del buf694  # reuse
        # Topologically Sorted Source Nodes: [features_65_conv_1, sigmoid_328, mul_198], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf695, buf693, primals_1044, primals_1045, primals_1046, primals_1047, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_65_conv_3], Original ATen: [aten.convolution]
        buf696 = extern_kernels.convolution(buf695, primals_1048, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf696, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf697 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf696, primals_1049, primals_1050, primals_1051, primals_1052, buf697, 49152, grid=grid(49152), stream=stream0)
        buf698 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_330, mul_199, features_65_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf697, buf698, 12288, grid=grid(12288), stream=stream0)
        buf699 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1054, reinterpret_tensor(buf698, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1053, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf699)
        del primals_1054
        buf700 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_332, mul_200], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf699, buf700, 512, grid=grid(512), stream=stream0)
        buf701 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1056, buf700, reinterpret_tensor(primals_1055, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf701)
        del primals_1056
        buf702 = buf697; del buf697  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_330, mul_199, mul_201], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf702, buf701, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_65_conv_7], Original ATen: [aten.convolution]
        buf703 = extern_kernels.convolution(buf702, primals_1057, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf703, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf704 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_65_conv_8, add_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf692, buf703, primals_1058, primals_1059, primals_1060, primals_1061, buf704, 8192, grid=grid(8192), stream=stream0)
        del primals_1061
        # Topologically Sorted Source Nodes: [features_66_conv_0], Original ATen: [aten.convolution]
        buf705 = extern_kernels.convolution(buf704, primals_1062, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf705, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf706 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf707 = buf706; del buf706  # reuse
        # Topologically Sorted Source Nodes: [features_66_conv_1, sigmoid_335, mul_202], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf707, buf705, primals_1063, primals_1064, primals_1065, primals_1066, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_66_conv_3], Original ATen: [aten.convolution]
        buf708 = extern_kernels.convolution(buf707, primals_1067, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf708, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf709 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf708, primals_1068, primals_1069, primals_1070, primals_1071, buf709, 49152, grid=grid(49152), stream=stream0)
        buf710 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_337, mul_203, features_66_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf709, buf710, 12288, grid=grid(12288), stream=stream0)
        buf711 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1073, reinterpret_tensor(buf710, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1072, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf711)
        del primals_1073
        buf712 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_339, mul_204], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf711, buf712, 512, grid=grid(512), stream=stream0)
        buf713 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1075, buf712, reinterpret_tensor(primals_1074, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf713)
        del primals_1075
        buf714 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_337, mul_203, mul_205], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf714, buf713, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_66_conv_7], Original ATen: [aten.convolution]
        buf715 = extern_kernels.convolution(buf714, primals_1076, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf715, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf716 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_66_conv_8, add_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf704, buf715, primals_1077, primals_1078, primals_1079, primals_1080, buf716, 8192, grid=grid(8192), stream=stream0)
        del primals_1080
        # Topologically Sorted Source Nodes: [features_67_conv_0], Original ATen: [aten.convolution]
        buf717 = extern_kernels.convolution(buf716, primals_1081, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf717, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf718 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf719 = buf718; del buf718  # reuse
        # Topologically Sorted Source Nodes: [features_67_conv_1, sigmoid_342, mul_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf719, buf717, primals_1082, primals_1083, primals_1084, primals_1085, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_67_conv_3], Original ATen: [aten.convolution]
        buf720 = extern_kernels.convolution(buf719, primals_1086, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf720, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf721 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf720, primals_1087, primals_1088, primals_1089, primals_1090, buf721, 49152, grid=grid(49152), stream=stream0)
        buf722 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_344, mul_207, features_67_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf721, buf722, 12288, grid=grid(12288), stream=stream0)
        buf723 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1092, reinterpret_tensor(buf722, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1091, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf723)
        del primals_1092
        buf724 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_346, mul_208], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf723, buf724, 512, grid=grid(512), stream=stream0)
        buf725 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1094, buf724, reinterpret_tensor(primals_1093, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf725)
        del primals_1094
        buf726 = buf721; del buf721  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_344, mul_207, mul_209], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf726, buf725, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_67_conv_7], Original ATen: [aten.convolution]
        buf727 = extern_kernels.convolution(buf726, primals_1095, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf727, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf728 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_67_conv_8, add_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf716, buf727, primals_1096, primals_1097, primals_1098, primals_1099, buf728, 8192, grid=grid(8192), stream=stream0)
        del primals_1099
        # Topologically Sorted Source Nodes: [features_68_conv_0], Original ATen: [aten.convolution]
        buf729 = extern_kernels.convolution(buf728, primals_1100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf729, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf730 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf731 = buf730; del buf730  # reuse
        # Topologically Sorted Source Nodes: [features_68_conv_1, sigmoid_349, mul_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf731, buf729, primals_1101, primals_1102, primals_1103, primals_1104, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_68_conv_3], Original ATen: [aten.convolution]
        buf732 = extern_kernels.convolution(buf731, primals_1105, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf732, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf733 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf732, primals_1106, primals_1107, primals_1108, primals_1109, buf733, 49152, grid=grid(49152), stream=stream0)
        buf734 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_351, mul_211, features_68_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf733, buf734, 12288, grid=grid(12288), stream=stream0)
        buf735 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1111, reinterpret_tensor(buf734, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1110, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf735)
        del primals_1111
        buf736 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_353, mul_212], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf735, buf736, 512, grid=grid(512), stream=stream0)
        buf737 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1113, buf736, reinterpret_tensor(primals_1112, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf737)
        del primals_1113
        buf738 = buf733; del buf733  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_351, mul_211, mul_213], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf738, buf737, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_68_conv_7], Original ATen: [aten.convolution]
        buf739 = extern_kernels.convolution(buf738, primals_1114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf739, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf740 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_68_conv_8, add_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf728, buf739, primals_1115, primals_1116, primals_1117, primals_1118, buf740, 8192, grid=grid(8192), stream=stream0)
        del primals_1118
        # Topologically Sorted Source Nodes: [features_69_conv_0], Original ATen: [aten.convolution]
        buf741 = extern_kernels.convolution(buf740, primals_1119, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf741, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf742 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf743 = buf742; del buf742  # reuse
        # Topologically Sorted Source Nodes: [features_69_conv_1, sigmoid_356, mul_214], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf743, buf741, primals_1120, primals_1121, primals_1122, primals_1123, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_69_conv_3], Original ATen: [aten.convolution]
        buf744 = extern_kernels.convolution(buf743, primals_1124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf744, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf745 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf744, primals_1125, primals_1126, primals_1127, primals_1128, buf745, 49152, grid=grid(49152), stream=stream0)
        buf746 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_358, mul_215, features_69_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf745, buf746, 12288, grid=grid(12288), stream=stream0)
        buf747 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1130, reinterpret_tensor(buf746, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1129, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf747)
        del primals_1130
        buf748 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_360, mul_216], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf747, buf748, 512, grid=grid(512), stream=stream0)
        buf749 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1132, buf748, reinterpret_tensor(primals_1131, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf749)
        del primals_1132
        buf750 = buf745; del buf745  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_358, mul_215, mul_217], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf750, buf749, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_69_conv_7], Original ATen: [aten.convolution]
        buf751 = extern_kernels.convolution(buf750, primals_1133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf751, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf752 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_69_conv_8, add_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf740, buf751, primals_1134, primals_1135, primals_1136, primals_1137, buf752, 8192, grid=grid(8192), stream=stream0)
        del primals_1137
        # Topologically Sorted Source Nodes: [features_70_conv_0], Original ATen: [aten.convolution]
        buf753 = extern_kernels.convolution(buf752, primals_1138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf753, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf754 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf755 = buf754; del buf754  # reuse
        # Topologically Sorted Source Nodes: [features_70_conv_1, sigmoid_363, mul_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf755, buf753, primals_1139, primals_1140, primals_1141, primals_1142, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_70_conv_3], Original ATen: [aten.convolution]
        buf756 = extern_kernels.convolution(buf755, primals_1143, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf756, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf757 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf756, primals_1144, primals_1145, primals_1146, primals_1147, buf757, 49152, grid=grid(49152), stream=stream0)
        buf758 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_365, mul_219, features_70_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf757, buf758, 12288, grid=grid(12288), stream=stream0)
        buf759 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1149, reinterpret_tensor(buf758, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1148, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf759)
        del primals_1149
        buf760 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_367, mul_220], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf759, buf760, 512, grid=grid(512), stream=stream0)
        buf761 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1151, buf760, reinterpret_tensor(primals_1150, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf761)
        del primals_1151
        buf762 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_365, mul_219, mul_221], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf762, buf761, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_70_conv_7], Original ATen: [aten.convolution]
        buf763 = extern_kernels.convolution(buf762, primals_1152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf763, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf764 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_70_conv_8, add_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf752, buf763, primals_1153, primals_1154, primals_1155, primals_1156, buf764, 8192, grid=grid(8192), stream=stream0)
        del primals_1156
        # Topologically Sorted Source Nodes: [features_71_conv_0], Original ATen: [aten.convolution]
        buf765 = extern_kernels.convolution(buf764, primals_1157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf765, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf766 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf767 = buf766; del buf766  # reuse
        # Topologically Sorted Source Nodes: [features_71_conv_1, sigmoid_370, mul_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf767, buf765, primals_1158, primals_1159, primals_1160, primals_1161, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_71_conv_3], Original ATen: [aten.convolution]
        buf768 = extern_kernels.convolution(buf767, primals_1162, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf768, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf769 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf768, primals_1163, primals_1164, primals_1165, primals_1166, buf769, 49152, grid=grid(49152), stream=stream0)
        buf770 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_372, mul_223, features_71_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf769, buf770, 12288, grid=grid(12288), stream=stream0)
        buf771 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1168, reinterpret_tensor(buf770, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1167, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf771)
        del primals_1168
        buf772 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_374, mul_224], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf771, buf772, 512, grid=grid(512), stream=stream0)
        buf773 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1170, buf772, reinterpret_tensor(primals_1169, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf773)
        del primals_1170
        buf774 = buf769; del buf769  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_372, mul_223, mul_225], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf774, buf773, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_71_conv_7], Original ATen: [aten.convolution]
        buf775 = extern_kernels.convolution(buf774, primals_1171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf775, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf776 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_71_conv_8, add_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf764, buf775, primals_1172, primals_1173, primals_1174, primals_1175, buf776, 8192, grid=grid(8192), stream=stream0)
        del primals_1175
        # Topologically Sorted Source Nodes: [features_72_conv_0], Original ATen: [aten.convolution]
        buf777 = extern_kernels.convolution(buf776, primals_1176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf777, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf778 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf779 = buf778; del buf778  # reuse
        # Topologically Sorted Source Nodes: [features_72_conv_1, sigmoid_377, mul_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf779, buf777, primals_1177, primals_1178, primals_1179, primals_1180, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_72_conv_3], Original ATen: [aten.convolution]
        buf780 = extern_kernels.convolution(buf779, primals_1181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf780, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf781 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf780, primals_1182, primals_1183, primals_1184, primals_1185, buf781, 49152, grid=grid(49152), stream=stream0)
        buf782 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_379, mul_227, features_72_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf781, buf782, 12288, grid=grid(12288), stream=stream0)
        buf783 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1187, reinterpret_tensor(buf782, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1186, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf783)
        del primals_1187
        buf784 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_381, mul_228], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf783, buf784, 512, grid=grid(512), stream=stream0)
        buf785 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1189, buf784, reinterpret_tensor(primals_1188, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf785)
        del primals_1189
        buf786 = buf781; del buf781  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_379, mul_227, mul_229], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf786, buf785, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_72_conv_7], Original ATen: [aten.convolution]
        buf787 = extern_kernels.convolution(buf786, primals_1190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf787, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf788 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_72_conv_8, add_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf776, buf787, primals_1191, primals_1192, primals_1193, primals_1194, buf788, 8192, grid=grid(8192), stream=stream0)
        del primals_1194
        # Topologically Sorted Source Nodes: [features_73_conv_0], Original ATen: [aten.convolution]
        buf789 = extern_kernels.convolution(buf788, primals_1195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf789, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf790 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf791 = buf790; del buf790  # reuse
        # Topologically Sorted Source Nodes: [features_73_conv_1, sigmoid_384, mul_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf791, buf789, primals_1196, primals_1197, primals_1198, primals_1199, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_73_conv_3], Original ATen: [aten.convolution]
        buf792 = extern_kernels.convolution(buf791, primals_1200, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf792, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf793 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf792, primals_1201, primals_1202, primals_1203, primals_1204, buf793, 49152, grid=grid(49152), stream=stream0)
        buf794 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_386, mul_231, features_73_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf793, buf794, 12288, grid=grid(12288), stream=stream0)
        buf795 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1206, reinterpret_tensor(buf794, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1205, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf795)
        del primals_1206
        buf796 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_388, mul_232], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf795, buf796, 512, grid=grid(512), stream=stream0)
        buf797 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1208, buf796, reinterpret_tensor(primals_1207, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf797)
        del primals_1208
        buf798 = buf793; del buf793  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_386, mul_231, mul_233], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf798, buf797, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_73_conv_7], Original ATen: [aten.convolution]
        buf799 = extern_kernels.convolution(buf798, primals_1209, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf799, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf800 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_73_conv_8, add_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf788, buf799, primals_1210, primals_1211, primals_1212, primals_1213, buf800, 8192, grid=grid(8192), stream=stream0)
        del primals_1213
        # Topologically Sorted Source Nodes: [features_74_conv_0], Original ATen: [aten.convolution]
        buf801 = extern_kernels.convolution(buf800, primals_1214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf801, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf802 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf803 = buf802; del buf802  # reuse
        # Topologically Sorted Source Nodes: [features_74_conv_1, sigmoid_391, mul_234], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf803, buf801, primals_1215, primals_1216, primals_1217, primals_1218, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_74_conv_3], Original ATen: [aten.convolution]
        buf804 = extern_kernels.convolution(buf803, primals_1219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf804, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf805 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf804, primals_1220, primals_1221, primals_1222, primals_1223, buf805, 49152, grid=grid(49152), stream=stream0)
        buf806 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_393, mul_235, features_74_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf805, buf806, 12288, grid=grid(12288), stream=stream0)
        buf807 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1225, reinterpret_tensor(buf806, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1224, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf807)
        del primals_1225
        buf808 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_395, mul_236], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf807, buf808, 512, grid=grid(512), stream=stream0)
        buf809 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1227, buf808, reinterpret_tensor(primals_1226, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf809)
        del primals_1227
        buf810 = buf805; del buf805  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_393, mul_235, mul_237], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf810, buf809, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_74_conv_7], Original ATen: [aten.convolution]
        buf811 = extern_kernels.convolution(buf810, primals_1228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf811, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf812 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_74_conv_8, add_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf800, buf811, primals_1229, primals_1230, primals_1231, primals_1232, buf812, 8192, grid=grid(8192), stream=stream0)
        del primals_1232
        # Topologically Sorted Source Nodes: [features_75_conv_0], Original ATen: [aten.convolution]
        buf813 = extern_kernels.convolution(buf812, primals_1233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf813, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf814 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf815 = buf814; del buf814  # reuse
        # Topologically Sorted Source Nodes: [features_75_conv_1, sigmoid_398, mul_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf815, buf813, primals_1234, primals_1235, primals_1236, primals_1237, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_75_conv_3], Original ATen: [aten.convolution]
        buf816 = extern_kernels.convolution(buf815, primals_1238, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf816, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf817 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf816, primals_1239, primals_1240, primals_1241, primals_1242, buf817, 49152, grid=grid(49152), stream=stream0)
        buf818 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_400, mul_239, features_75_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf817, buf818, 12288, grid=grid(12288), stream=stream0)
        buf819 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1244, reinterpret_tensor(buf818, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1243, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf819)
        del primals_1244
        buf820 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_402, mul_240], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf819, buf820, 512, grid=grid(512), stream=stream0)
        buf821 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1246, buf820, reinterpret_tensor(primals_1245, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf821)
        del primals_1246
        buf822 = buf817; del buf817  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_400, mul_239, mul_241], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf822, buf821, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_75_conv_7], Original ATen: [aten.convolution]
        buf823 = extern_kernels.convolution(buf822, primals_1247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf823, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf824 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_75_conv_8, add_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf812, buf823, primals_1248, primals_1249, primals_1250, primals_1251, buf824, 8192, grid=grid(8192), stream=stream0)
        del primals_1251
        # Topologically Sorted Source Nodes: [features_76_conv_0], Original ATen: [aten.convolution]
        buf825 = extern_kernels.convolution(buf824, primals_1252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf825, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf826 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf827 = buf826; del buf826  # reuse
        # Topologically Sorted Source Nodes: [features_76_conv_1, sigmoid_405, mul_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf827, buf825, primals_1253, primals_1254, primals_1255, primals_1256, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_76_conv_3], Original ATen: [aten.convolution]
        buf828 = extern_kernels.convolution(buf827, primals_1257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf828, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf829 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf828, primals_1258, primals_1259, primals_1260, primals_1261, buf829, 49152, grid=grid(49152), stream=stream0)
        buf830 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_407, mul_243, features_76_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf829, buf830, 12288, grid=grid(12288), stream=stream0)
        buf831 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1263, reinterpret_tensor(buf830, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1262, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf831)
        del primals_1263
        buf832 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_409, mul_244], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf831, buf832, 512, grid=grid(512), stream=stream0)
        buf833 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1265, buf832, reinterpret_tensor(primals_1264, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf833)
        del primals_1265
        buf834 = buf829; del buf829  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_407, mul_243, mul_245], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf834, buf833, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_76_conv_7], Original ATen: [aten.convolution]
        buf835 = extern_kernels.convolution(buf834, primals_1266, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf835, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf836 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_76_conv_8, add_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf824, buf835, primals_1267, primals_1268, primals_1269, primals_1270, buf836, 8192, grid=grid(8192), stream=stream0)
        del primals_1270
        # Topologically Sorted Source Nodes: [features_77_conv_0], Original ATen: [aten.convolution]
        buf837 = extern_kernels.convolution(buf836, primals_1271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf837, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf838 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf839 = buf838; del buf838  # reuse
        # Topologically Sorted Source Nodes: [features_77_conv_1, sigmoid_412, mul_246], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf839, buf837, primals_1272, primals_1273, primals_1274, primals_1275, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_77_conv_3], Original ATen: [aten.convolution]
        buf840 = extern_kernels.convolution(buf839, primals_1276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf840, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf841 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf840, primals_1277, primals_1278, primals_1279, primals_1280, buf841, 49152, grid=grid(49152), stream=stream0)
        buf842 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_414, mul_247, features_77_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf841, buf842, 12288, grid=grid(12288), stream=stream0)
        buf843 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1282, reinterpret_tensor(buf842, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1281, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf843)
        del primals_1282
        buf844 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_416, mul_248], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf843, buf844, 512, grid=grid(512), stream=stream0)
        buf845 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1284, buf844, reinterpret_tensor(primals_1283, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf845)
        del primals_1284
        buf846 = buf841; del buf841  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_414, mul_247, mul_249], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf846, buf845, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_77_conv_7], Original ATen: [aten.convolution]
        buf847 = extern_kernels.convolution(buf846, primals_1285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf847, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf848 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_77_conv_8, add_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf836, buf847, primals_1286, primals_1287, primals_1288, primals_1289, buf848, 8192, grid=grid(8192), stream=stream0)
        del primals_1289
        # Topologically Sorted Source Nodes: [features_78_conv_0], Original ATen: [aten.convolution]
        buf849 = extern_kernels.convolution(buf848, primals_1290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf849, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf850 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf851 = buf850; del buf850  # reuse
        # Topologically Sorted Source Nodes: [features_78_conv_1, sigmoid_419, mul_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf851, buf849, primals_1291, primals_1292, primals_1293, primals_1294, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_78_conv_3], Original ATen: [aten.convolution]
        buf852 = extern_kernels.convolution(buf851, primals_1295, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf852, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf853 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf852, primals_1296, primals_1297, primals_1298, primals_1299, buf853, 49152, grid=grid(49152), stream=stream0)
        buf854 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_421, mul_251, features_78_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf853, buf854, 12288, grid=grid(12288), stream=stream0)
        buf855 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1301, reinterpret_tensor(buf854, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1300, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf855)
        del primals_1301
        buf856 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_423, mul_252], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf855, buf856, 512, grid=grid(512), stream=stream0)
        buf857 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1303, buf856, reinterpret_tensor(primals_1302, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf857)
        del primals_1303
        buf858 = buf853; del buf853  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_421, mul_251, mul_253], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf858, buf857, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_78_conv_7], Original ATen: [aten.convolution]
        buf859 = extern_kernels.convolution(buf858, primals_1304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf859, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf860 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_78_conv_8, add_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf848, buf859, primals_1305, primals_1306, primals_1307, primals_1308, buf860, 8192, grid=grid(8192), stream=stream0)
        del primals_1308
        # Topologically Sorted Source Nodes: [features_79_conv_0], Original ATen: [aten.convolution]
        buf861 = extern_kernels.convolution(buf860, primals_1309, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf861, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf862 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf863 = buf862; del buf862  # reuse
        # Topologically Sorted Source Nodes: [features_79_conv_1, sigmoid_426, mul_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf863, buf861, primals_1310, primals_1311, primals_1312, primals_1313, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_79_conv_3], Original ATen: [aten.convolution]
        buf864 = extern_kernels.convolution(buf863, primals_1314, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf864, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf865 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf864, primals_1315, primals_1316, primals_1317, primals_1318, buf865, 49152, grid=grid(49152), stream=stream0)
        buf866 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_428, mul_255, features_79_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf865, buf866, 12288, grid=grid(12288), stream=stream0)
        buf867 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1320, reinterpret_tensor(buf866, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1319, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf867)
        del primals_1320
        buf868 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_430, mul_256], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf867, buf868, 512, grid=grid(512), stream=stream0)
        buf869 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1322, buf868, reinterpret_tensor(primals_1321, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf869)
        del primals_1322
        buf870 = buf865; del buf865  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_428, mul_255, mul_257], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf870, buf869, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_79_conv_7], Original ATen: [aten.convolution]
        buf871 = extern_kernels.convolution(buf870, primals_1323, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf871, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf872 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_79_conv_8, add_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf860, buf871, primals_1324, primals_1325, primals_1326, primals_1327, buf872, 8192, grid=grid(8192), stream=stream0)
        del primals_1327
        # Topologically Sorted Source Nodes: [features_80_conv_0], Original ATen: [aten.convolution]
        buf873 = extern_kernels.convolution(buf872, primals_1328, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf873, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf874 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf875 = buf874; del buf874  # reuse
        # Topologically Sorted Source Nodes: [features_80_conv_1, sigmoid_433, mul_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf875, buf873, primals_1329, primals_1330, primals_1331, primals_1332, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_80_conv_3], Original ATen: [aten.convolution]
        buf876 = extern_kernels.convolution(buf875, primals_1333, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf876, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf877 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_80_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf876, primals_1334, primals_1335, primals_1336, primals_1337, buf877, 49152, grid=grid(49152), stream=stream0)
        buf878 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_435, mul_259, features_80_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf877, buf878, 12288, grid=grid(12288), stream=stream0)
        buf879 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_80_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1339, reinterpret_tensor(buf878, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1338, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf879)
        del primals_1339
        buf880 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_437, mul_260], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf879, buf880, 512, grid=grid(512), stream=stream0)
        buf881 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_80_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1341, buf880, reinterpret_tensor(primals_1340, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf881)
        del primals_1341
        buf882 = buf877; del buf877  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_435, mul_259, mul_261], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf882, buf881, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_80_conv_7], Original ATen: [aten.convolution]
        buf883 = extern_kernels.convolution(buf882, primals_1342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf883, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf884 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_80_conv_8, add_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf872, buf883, primals_1343, primals_1344, primals_1345, primals_1346, buf884, 8192, grid=grid(8192), stream=stream0)
        del primals_1346
        # Topologically Sorted Source Nodes: [features_81_conv_0], Original ATen: [aten.convolution]
        buf885 = extern_kernels.convolution(buf884, primals_1347, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf885, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf886 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf887 = buf886; del buf886  # reuse
        # Topologically Sorted Source Nodes: [features_81_conv_1, sigmoid_440, mul_262], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf887, buf885, primals_1348, primals_1349, primals_1350, primals_1351, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_81_conv_3], Original ATen: [aten.convolution]
        buf888 = extern_kernels.convolution(buf887, primals_1352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf888, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf889 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_81_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf888, primals_1353, primals_1354, primals_1355, primals_1356, buf889, 49152, grid=grid(49152), stream=stream0)
        buf890 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_442, mul_263, features_81_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf889, buf890, 12288, grid=grid(12288), stream=stream0)
        buf891 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_81_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1358, reinterpret_tensor(buf890, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1357, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf891)
        del primals_1358
        buf892 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_444, mul_264], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf891, buf892, 512, grid=grid(512), stream=stream0)
        buf893 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_81_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1360, buf892, reinterpret_tensor(primals_1359, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf893)
        del primals_1360
        buf894 = buf889; del buf889  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_442, mul_263, mul_265], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf894, buf893, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_81_conv_7], Original ATen: [aten.convolution]
        buf895 = extern_kernels.convolution(buf894, primals_1361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf895, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf896 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_81_conv_8, add_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf884, buf895, primals_1362, primals_1363, primals_1364, primals_1365, buf896, 8192, grid=grid(8192), stream=stream0)
        del primals_1365
        # Topologically Sorted Source Nodes: [features_82_conv_0], Original ATen: [aten.convolution]
        buf897 = extern_kernels.convolution(buf896, primals_1366, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf897, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf898 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf899 = buf898; del buf898  # reuse
        # Topologically Sorted Source Nodes: [features_82_conv_1, sigmoid_447, mul_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf899, buf897, primals_1367, primals_1368, primals_1369, primals_1370, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_82_conv_3], Original ATen: [aten.convolution]
        buf900 = extern_kernels.convolution(buf899, primals_1371, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf900, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf901 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_82_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf900, primals_1372, primals_1373, primals_1374, primals_1375, buf901, 49152, grid=grid(49152), stream=stream0)
        buf902 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_449, mul_267, features_82_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf901, buf902, 12288, grid=grid(12288), stream=stream0)
        buf903 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_82_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1377, reinterpret_tensor(buf902, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1376, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf903)
        del primals_1377
        buf904 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_451, mul_268], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf903, buf904, 512, grid=grid(512), stream=stream0)
        buf905 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_82_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1379, buf904, reinterpret_tensor(primals_1378, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf905)
        del primals_1379
        buf906 = buf901; del buf901  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_449, mul_267, mul_269], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf906, buf905, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_82_conv_7], Original ATen: [aten.convolution]
        buf907 = extern_kernels.convolution(buf906, primals_1380, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf907, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf908 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_82_conv_8, add_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf896, buf907, primals_1381, primals_1382, primals_1383, primals_1384, buf908, 8192, grid=grid(8192), stream=stream0)
        del primals_1384
        # Topologically Sorted Source Nodes: [features_83_conv_0], Original ATen: [aten.convolution]
        buf909 = extern_kernels.convolution(buf908, primals_1385, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf909, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf910 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf911 = buf910; del buf910  # reuse
        # Topologically Sorted Source Nodes: [features_83_conv_1, sigmoid_454, mul_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf911, buf909, primals_1386, primals_1387, primals_1388, primals_1389, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_83_conv_3], Original ATen: [aten.convolution]
        buf912 = extern_kernels.convolution(buf911, primals_1390, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf912, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf913 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_83_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf912, primals_1391, primals_1392, primals_1393, primals_1394, buf913, 49152, grid=grid(49152), stream=stream0)
        buf914 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_456, mul_271, features_83_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf913, buf914, 12288, grid=grid(12288), stream=stream0)
        buf915 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_83_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1396, reinterpret_tensor(buf914, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1395, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf915)
        del primals_1396
        buf916 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_458, mul_272], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf915, buf916, 512, grid=grid(512), stream=stream0)
        buf917 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_83_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1398, buf916, reinterpret_tensor(primals_1397, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf917)
        del primals_1398
        buf918 = buf913; del buf913  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_456, mul_271, mul_273], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf918, buf917, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_83_conv_7], Original ATen: [aten.convolution]
        buf919 = extern_kernels.convolution(buf918, primals_1399, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf919, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf920 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_83_conv_8, add_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf908, buf919, primals_1400, primals_1401, primals_1402, primals_1403, buf920, 8192, grid=grid(8192), stream=stream0)
        del primals_1403
        # Topologically Sorted Source Nodes: [features_84_conv_0], Original ATen: [aten.convolution]
        buf921 = extern_kernels.convolution(buf920, primals_1404, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf921, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf922 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf923 = buf922; del buf922  # reuse
        # Topologically Sorted Source Nodes: [features_84_conv_1, sigmoid_461, mul_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf923, buf921, primals_1405, primals_1406, primals_1407, primals_1408, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_84_conv_3], Original ATen: [aten.convolution]
        buf924 = extern_kernels.convolution(buf923, primals_1409, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf924, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf925 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_84_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf924, primals_1410, primals_1411, primals_1412, primals_1413, buf925, 49152, grid=grid(49152), stream=stream0)
        buf926 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_463, mul_275, features_84_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf925, buf926, 12288, grid=grid(12288), stream=stream0)
        buf927 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_84_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1415, reinterpret_tensor(buf926, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1414, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf927)
        del primals_1415
        buf928 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_465, mul_276], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf927, buf928, 512, grid=grid(512), stream=stream0)
        buf929 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_84_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1417, buf928, reinterpret_tensor(primals_1416, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf929)
        del primals_1417
        buf930 = buf925; del buf925  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_463, mul_275, mul_277], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf930, buf929, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_84_conv_7], Original ATen: [aten.convolution]
        buf931 = extern_kernels.convolution(buf930, primals_1418, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf931, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf932 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_84_conv_8, add_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf920, buf931, primals_1419, primals_1420, primals_1421, primals_1422, buf932, 8192, grid=grid(8192), stream=stream0)
        del primals_1422
        # Topologically Sorted Source Nodes: [features_85_conv_0], Original ATen: [aten.convolution]
        buf933 = extern_kernels.convolution(buf932, primals_1423, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf933, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf934 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf935 = buf934; del buf934  # reuse
        # Topologically Sorted Source Nodes: [features_85_conv_1, sigmoid_468, mul_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf935, buf933, primals_1424, primals_1425, primals_1426, primals_1427, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_85_conv_3], Original ATen: [aten.convolution]
        buf936 = extern_kernels.convolution(buf935, primals_1428, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf936, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf937 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_85_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf936, primals_1429, primals_1430, primals_1431, primals_1432, buf937, 49152, grid=grid(49152), stream=stream0)
        buf938 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_470, mul_279, features_85_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf937, buf938, 12288, grid=grid(12288), stream=stream0)
        buf939 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_85_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1434, reinterpret_tensor(buf938, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1433, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf939)
        del primals_1434
        buf940 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_472, mul_280], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf939, buf940, 512, grid=grid(512), stream=stream0)
        buf941 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_85_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1436, buf940, reinterpret_tensor(primals_1435, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf941)
        del primals_1436
        buf942 = buf937; del buf937  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_470, mul_279, mul_281], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf942, buf941, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_85_conv_7], Original ATen: [aten.convolution]
        buf943 = extern_kernels.convolution(buf942, primals_1437, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf943, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf944 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_85_conv_8, add_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf932, buf943, primals_1438, primals_1439, primals_1440, primals_1441, buf944, 8192, grid=grid(8192), stream=stream0)
        del primals_1441
        # Topologically Sorted Source Nodes: [features_86_conv_0], Original ATen: [aten.convolution]
        buf945 = extern_kernels.convolution(buf944, primals_1442, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf945, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf946 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf947 = buf946; del buf946  # reuse
        # Topologically Sorted Source Nodes: [features_86_conv_1, sigmoid_475, mul_282], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf947, buf945, primals_1443, primals_1444, primals_1445, primals_1446, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_86_conv_3], Original ATen: [aten.convolution]
        buf948 = extern_kernels.convolution(buf947, primals_1447, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf948, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf949 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_86_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf948, primals_1448, primals_1449, primals_1450, primals_1451, buf949, 49152, grid=grid(49152), stream=stream0)
        buf950 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_477, mul_283, features_86_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf949, buf950, 12288, grid=grid(12288), stream=stream0)
        buf951 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_86_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1453, reinterpret_tensor(buf950, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1452, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf951)
        del primals_1453
        buf952 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_479, mul_284], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf951, buf952, 512, grid=grid(512), stream=stream0)
        buf953 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_86_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1455, buf952, reinterpret_tensor(primals_1454, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf953)
        del primals_1455
        buf954 = buf949; del buf949  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_477, mul_283, mul_285], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf954, buf953, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_86_conv_7], Original ATen: [aten.convolution]
        buf955 = extern_kernels.convolution(buf954, primals_1456, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf955, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf956 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_86_conv_8, add_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf944, buf955, primals_1457, primals_1458, primals_1459, primals_1460, buf956, 8192, grid=grid(8192), stream=stream0)
        del primals_1460
        # Topologically Sorted Source Nodes: [features_87_conv_0], Original ATen: [aten.convolution]
        buf957 = extern_kernels.convolution(buf956, primals_1461, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf957, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf958 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf959 = buf958; del buf958  # reuse
        # Topologically Sorted Source Nodes: [features_87_conv_1, sigmoid_482, mul_286], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf959, buf957, primals_1462, primals_1463, primals_1464, primals_1465, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_87_conv_3], Original ATen: [aten.convolution]
        buf960 = extern_kernels.convolution(buf959, primals_1466, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf960, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf961 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_87_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf960, primals_1467, primals_1468, primals_1469, primals_1470, buf961, 49152, grid=grid(49152), stream=stream0)
        buf962 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_484, mul_287, features_87_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf961, buf962, 12288, grid=grid(12288), stream=stream0)
        buf963 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_87_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1472, reinterpret_tensor(buf962, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1471, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf963)
        del primals_1472
        buf964 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_486, mul_288], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf963, buf964, 512, grid=grid(512), stream=stream0)
        buf965 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_87_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1474, buf964, reinterpret_tensor(primals_1473, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf965)
        del primals_1474
        buf966 = buf961; del buf961  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_484, mul_287, mul_289], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf966, buf965, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_87_conv_7], Original ATen: [aten.convolution]
        buf967 = extern_kernels.convolution(buf966, primals_1475, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf967, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf968 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_87_conv_8, add_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf956, buf967, primals_1476, primals_1477, primals_1478, primals_1479, buf968, 8192, grid=grid(8192), stream=stream0)
        del primals_1479
        # Topologically Sorted Source Nodes: [features_88_conv_0], Original ATen: [aten.convolution]
        buf969 = extern_kernels.convolution(buf968, primals_1480, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf969, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf970 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf971 = buf970; del buf970  # reuse
        # Topologically Sorted Source Nodes: [features_88_conv_1, sigmoid_489, mul_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf971, buf969, primals_1481, primals_1482, primals_1483, primals_1484, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_88_conv_3], Original ATen: [aten.convolution]
        buf972 = extern_kernels.convolution(buf971, primals_1485, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf972, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf973 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_88_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf972, primals_1486, primals_1487, primals_1488, primals_1489, buf973, 49152, grid=grid(49152), stream=stream0)
        buf974 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_491, mul_291, features_88_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf973, buf974, 12288, grid=grid(12288), stream=stream0)
        buf975 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_88_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1491, reinterpret_tensor(buf974, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1490, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf975)
        del primals_1491
        buf976 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_493, mul_292], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf975, buf976, 512, grid=grid(512), stream=stream0)
        buf977 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_88_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1493, buf976, reinterpret_tensor(primals_1492, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf977)
        del primals_1493
        buf978 = buf973; del buf973  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_491, mul_291, mul_293], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf978, buf977, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_88_conv_7], Original ATen: [aten.convolution]
        buf979 = extern_kernels.convolution(buf978, primals_1494, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf979, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf980 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_88_conv_8, add_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf968, buf979, primals_1495, primals_1496, primals_1497, primals_1498, buf980, 8192, grid=grid(8192), stream=stream0)
        del primals_1498
        # Topologically Sorted Source Nodes: [features_89_conv_0], Original ATen: [aten.convolution]
        buf981 = extern_kernels.convolution(buf980, primals_1499, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf981, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf982 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf983 = buf982; del buf982  # reuse
        # Topologically Sorted Source Nodes: [features_89_conv_1, sigmoid_496, mul_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf983, buf981, primals_1500, primals_1501, primals_1502, primals_1503, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_89_conv_3], Original ATen: [aten.convolution]
        buf984 = extern_kernels.convolution(buf983, primals_1504, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf984, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf985 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_89_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf984, primals_1505, primals_1506, primals_1507, primals_1508, buf985, 49152, grid=grid(49152), stream=stream0)
        buf986 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_498, mul_295, features_89_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf985, buf986, 12288, grid=grid(12288), stream=stream0)
        buf987 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_89_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1510, reinterpret_tensor(buf986, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1509, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf987)
        del primals_1510
        buf988 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_500, mul_296], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf987, buf988, 512, grid=grid(512), stream=stream0)
        buf989 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_89_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1512, buf988, reinterpret_tensor(primals_1511, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf989)
        del primals_1512
        buf990 = buf985; del buf985  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_498, mul_295, mul_297], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf990, buf989, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_89_conv_7], Original ATen: [aten.convolution]
        buf991 = extern_kernels.convolution(buf990, primals_1513, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf991, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf992 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_89_conv_8, add_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf980, buf991, primals_1514, primals_1515, primals_1516, primals_1517, buf992, 8192, grid=grid(8192), stream=stream0)
        del primals_1517
        # Topologically Sorted Source Nodes: [features_90_conv_0], Original ATen: [aten.convolution]
        buf993 = extern_kernels.convolution(buf992, primals_1518, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf993, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf994 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf995 = buf994; del buf994  # reuse
        # Topologically Sorted Source Nodes: [features_90_conv_1, sigmoid_503, mul_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf995, buf993, primals_1519, primals_1520, primals_1521, primals_1522, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_90_conv_3], Original ATen: [aten.convolution]
        buf996 = extern_kernels.convolution(buf995, primals_1523, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf996, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf997 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_90_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf996, primals_1524, primals_1525, primals_1526, primals_1527, buf997, 49152, grid=grid(49152), stream=stream0)
        buf998 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_505, mul_299, features_90_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf997, buf998, 12288, grid=grid(12288), stream=stream0)
        buf999 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_90_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1529, reinterpret_tensor(buf998, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1528, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf999)
        del primals_1529
        buf1000 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_507, mul_300], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf999, buf1000, 512, grid=grid(512), stream=stream0)
        buf1001 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_90_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1531, buf1000, reinterpret_tensor(primals_1530, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf1001)
        del primals_1531
        buf1002 = buf997; del buf997  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_505, mul_299, mul_301], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf1002, buf1001, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_90_conv_7], Original ATen: [aten.convolution]
        buf1003 = extern_kernels.convolution(buf1002, primals_1532, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1003, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf1004 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_90_conv_8, add_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf992, buf1003, primals_1533, primals_1534, primals_1535, primals_1536, buf1004, 8192, grid=grid(8192), stream=stream0)
        del primals_1536
        # Topologically Sorted Source Nodes: [features_91_conv_0], Original ATen: [aten.convolution]
        buf1005 = extern_kernels.convolution(buf1004, primals_1537, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1005, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf1006 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf1007 = buf1006; del buf1006  # reuse
        # Topologically Sorted Source Nodes: [features_91_conv_1, sigmoid_510, mul_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf1007, buf1005, primals_1538, primals_1539, primals_1540, primals_1541, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_91_conv_3], Original ATen: [aten.convolution]
        buf1008 = extern_kernels.convolution(buf1007, primals_1542, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf1008, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf1009 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_91_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf1008, primals_1543, primals_1544, primals_1545, primals_1546, buf1009, 49152, grid=grid(49152), stream=stream0)
        buf1010 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_512, mul_303, features_91_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf1009, buf1010, 12288, grid=grid(12288), stream=stream0)
        buf1011 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_91_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1548, reinterpret_tensor(buf1010, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1547, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf1011)
        del primals_1548
        buf1012 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_514, mul_304], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf1011, buf1012, 512, grid=grid(512), stream=stream0)
        buf1013 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_91_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1550, buf1012, reinterpret_tensor(primals_1549, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf1013)
        del primals_1550
        buf1014 = buf1009; del buf1009  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_512, mul_303, mul_305], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf1014, buf1013, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_91_conv_7], Original ATen: [aten.convolution]
        buf1015 = extern_kernels.convolution(buf1014, primals_1551, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1015, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf1016 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_91_conv_8, add_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf1004, buf1015, primals_1552, primals_1553, primals_1554, primals_1555, buf1016, 8192, grid=grid(8192), stream=stream0)
        del primals_1555
        # Topologically Sorted Source Nodes: [features_92_conv_0], Original ATen: [aten.convolution]
        buf1017 = extern_kernels.convolution(buf1016, primals_1556, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1017, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf1018 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf1019 = buf1018; del buf1018  # reuse
        # Topologically Sorted Source Nodes: [features_92_conv_1, sigmoid_517, mul_306], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf1019, buf1017, primals_1557, primals_1558, primals_1559, primals_1560, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_92_conv_3], Original ATen: [aten.convolution]
        buf1020 = extern_kernels.convolution(buf1019, primals_1561, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf1020, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf1021 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_92_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf1020, primals_1562, primals_1563, primals_1564, primals_1565, buf1021, 49152, grid=grid(49152), stream=stream0)
        buf1022 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_519, mul_307, features_92_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf1021, buf1022, 12288, grid=grid(12288), stream=stream0)
        buf1023 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_92_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1567, reinterpret_tensor(buf1022, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1566, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf1023)
        del primals_1567
        buf1024 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_521, mul_308], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf1023, buf1024, 512, grid=grid(512), stream=stream0)
        buf1025 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_92_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1569, buf1024, reinterpret_tensor(primals_1568, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf1025)
        del primals_1569
        buf1026 = buf1021; del buf1021  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_519, mul_307, mul_309], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf1026, buf1025, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_92_conv_7], Original ATen: [aten.convolution]
        buf1027 = extern_kernels.convolution(buf1026, primals_1570, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1027, (4, 512, 2, 2), (2048, 1, 1024, 512))
        buf1028 = empty_strided_cuda((4, 512, 2, 2), (2048, 1, 1024, 512), torch.float32)
        # Topologically Sorted Source Nodes: [features_92_conv_8, add_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_50.run(buf1016, buf1027, primals_1571, primals_1572, primals_1573, primals_1574, buf1028, 8192, grid=grid(8192), stream=stream0)
        del primals_1574
        # Topologically Sorted Source Nodes: [features_93_conv_0], Original ATen: [aten.convolution]
        buf1029 = extern_kernels.convolution(buf1028, primals_1575, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1029, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf1030 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        buf1031 = buf1030; del buf1030  # reuse
        # Topologically Sorted Source Nodes: [features_93_conv_1, sigmoid_524, mul_310], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_45.run(buf1031, buf1029, primals_1576, primals_1577, primals_1578, primals_1579, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_93_conv_3], Original ATen: [aten.convolution]
        buf1032 = extern_kernels.convolution(buf1031, primals_1580, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3072, bias=None)
        assert_size_stride(buf1032, (4, 3072, 2, 2), (12288, 1, 6144, 3072))
        buf1033 = empty_strided_cuda((4, 3072, 2, 2), (12288, 1, 6144, 3072), torch.float32)
        # Topologically Sorted Source Nodes: [features_93_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_46.run(buf1032, primals_1581, primals_1582, primals_1583, primals_1584, buf1033, 49152, grid=grid(49152), stream=stream0)
        buf1034 = empty_strided_cuda((4, 3072, 1, 1), (3072, 1, 12288, 12288), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_526, mul_311, features_93_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_47.run(buf1033, buf1034, 12288, grid=grid(12288), stream=stream0)
        buf1035 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_93_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1586, reinterpret_tensor(buf1034, (4, 3072), (3072, 1), 0), reinterpret_tensor(primals_1585, (3072, 128), (1, 3072), 0), alpha=1, beta=1, out=buf1035)
        del primals_1586
        buf1036 = empty_strided_cuda((4, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_528, mul_312], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_48.run(buf1035, buf1036, 512, grid=grid(512), stream=stream0)
        buf1037 = empty_strided_cuda((4, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_93_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1588, buf1036, reinterpret_tensor(primals_1587, (128, 3072), (1, 128), 0), alpha=1, beta=1, out=buf1037)
        del primals_1588
        buf1038 = buf1033; del buf1033  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_526, mul_311, mul_313], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_49.run(buf1038, buf1037, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [features_93_conv_7], Original ATen: [aten.convolution]
        buf1039 = extern_kernels.convolution(buf1038, primals_1589, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1039, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1040 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_93_conv_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_51.run(buf1039, primals_1590, primals_1591, primals_1592, primals_1593, buf1040, 10240, grid=grid(10240), stream=stream0)
        del primals_1593
        # Topologically Sorted Source Nodes: [features_94_conv_0], Original ATen: [aten.convolution]
        buf1041 = extern_kernels.convolution(buf1040, primals_1594, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1041, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1042 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf1043 = buf1042; del buf1042  # reuse
        # Topologically Sorted Source Nodes: [features_94_conv_1, sigmoid_531, mul_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf1043, buf1041, primals_1595, primals_1596, primals_1597, primals_1598, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_94_conv_3], Original ATen: [aten.convolution]
        buf1044 = extern_kernels.convolution(buf1043, primals_1599, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf1044, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1045 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_94_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf1044, primals_1600, primals_1601, primals_1602, primals_1603, buf1045, 61440, grid=grid(61440), stream=stream0)
        buf1046 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_533, mul_315, features_94_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf1045, buf1046, 15360, grid=grid(15360), stream=stream0)
        buf1047 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_94_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1605, reinterpret_tensor(buf1046, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1604, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf1047)
        del primals_1605
        buf1048 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_535, mul_316], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf1047, buf1048, 640, grid=grid(640), stream=stream0)
        buf1049 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_94_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1607, buf1048, reinterpret_tensor(primals_1606, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf1049)
        del primals_1607
        buf1050 = buf1045; del buf1045  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_533, mul_315, mul_317], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf1050, buf1049, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_94_conv_7], Original ATen: [aten.convolution]
        buf1051 = extern_kernels.convolution(buf1050, primals_1608, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1051, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1052 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_94_conv_8, add_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf1040, buf1051, primals_1609, primals_1610, primals_1611, primals_1612, buf1052, 10240, grid=grid(10240), stream=stream0)
        del primals_1612
        # Topologically Sorted Source Nodes: [features_95_conv_0], Original ATen: [aten.convolution]
        buf1053 = extern_kernels.convolution(buf1052, primals_1613, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1053, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1054 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf1055 = buf1054; del buf1054  # reuse
        # Topologically Sorted Source Nodes: [features_95_conv_1, sigmoid_538, mul_318], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf1055, buf1053, primals_1614, primals_1615, primals_1616, primals_1617, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_95_conv_3], Original ATen: [aten.convolution]
        buf1056 = extern_kernels.convolution(buf1055, primals_1618, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf1056, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1057 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_95_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf1056, primals_1619, primals_1620, primals_1621, primals_1622, buf1057, 61440, grid=grid(61440), stream=stream0)
        buf1058 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_540, mul_319, features_95_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf1057, buf1058, 15360, grid=grid(15360), stream=stream0)
        buf1059 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_95_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1624, reinterpret_tensor(buf1058, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1623, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf1059)
        del primals_1624
        buf1060 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_542, mul_320], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf1059, buf1060, 640, grid=grid(640), stream=stream0)
        buf1061 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_95_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1626, buf1060, reinterpret_tensor(primals_1625, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf1061)
        del primals_1626
        buf1062 = buf1057; del buf1057  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_540, mul_319, mul_321], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf1062, buf1061, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_95_conv_7], Original ATen: [aten.convolution]
        buf1063 = extern_kernels.convolution(buf1062, primals_1627, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1063, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1064 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_95_conv_8, add_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf1052, buf1063, primals_1628, primals_1629, primals_1630, primals_1631, buf1064, 10240, grid=grid(10240), stream=stream0)
        del primals_1631
        # Topologically Sorted Source Nodes: [features_96_conv_0], Original ATen: [aten.convolution]
        buf1065 = extern_kernels.convolution(buf1064, primals_1632, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1065, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1066 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf1067 = buf1066; del buf1066  # reuse
        # Topologically Sorted Source Nodes: [features_96_conv_1, sigmoid_545, mul_322], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf1067, buf1065, primals_1633, primals_1634, primals_1635, primals_1636, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_96_conv_3], Original ATen: [aten.convolution]
        buf1068 = extern_kernels.convolution(buf1067, primals_1637, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf1068, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1069 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_96_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf1068, primals_1638, primals_1639, primals_1640, primals_1641, buf1069, 61440, grid=grid(61440), stream=stream0)
        buf1070 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_547, mul_323, features_96_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf1069, buf1070, 15360, grid=grid(15360), stream=stream0)
        buf1071 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_96_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1643, reinterpret_tensor(buf1070, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1642, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf1071)
        del primals_1643
        buf1072 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_549, mul_324], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf1071, buf1072, 640, grid=grid(640), stream=stream0)
        buf1073 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_96_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1645, buf1072, reinterpret_tensor(primals_1644, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf1073)
        del primals_1645
        buf1074 = buf1069; del buf1069  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_547, mul_323, mul_325], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf1074, buf1073, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_96_conv_7], Original ATen: [aten.convolution]
        buf1075 = extern_kernels.convolution(buf1074, primals_1646, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1075, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1076 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_96_conv_8, add_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf1064, buf1075, primals_1647, primals_1648, primals_1649, primals_1650, buf1076, 10240, grid=grid(10240), stream=stream0)
        del primals_1650
        # Topologically Sorted Source Nodes: [features_97_conv_0], Original ATen: [aten.convolution]
        buf1077 = extern_kernels.convolution(buf1076, primals_1651, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1077, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1078 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf1079 = buf1078; del buf1078  # reuse
        # Topologically Sorted Source Nodes: [features_97_conv_1, sigmoid_552, mul_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf1079, buf1077, primals_1652, primals_1653, primals_1654, primals_1655, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_97_conv_3], Original ATen: [aten.convolution]
        buf1080 = extern_kernels.convolution(buf1079, primals_1656, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf1080, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1081 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_97_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf1080, primals_1657, primals_1658, primals_1659, primals_1660, buf1081, 61440, grid=grid(61440), stream=stream0)
        buf1082 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_554, mul_327, features_97_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf1081, buf1082, 15360, grid=grid(15360), stream=stream0)
        buf1083 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_97_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1662, reinterpret_tensor(buf1082, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1661, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf1083)
        del primals_1662
        buf1084 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_556, mul_328], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf1083, buf1084, 640, grid=grid(640), stream=stream0)
        buf1085 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_97_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1664, buf1084, reinterpret_tensor(primals_1663, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf1085)
        del primals_1664
        buf1086 = buf1081; del buf1081  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_554, mul_327, mul_329], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf1086, buf1085, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_97_conv_7], Original ATen: [aten.convolution]
        buf1087 = extern_kernels.convolution(buf1086, primals_1665, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1087, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1088 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_97_conv_8, add_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf1076, buf1087, primals_1666, primals_1667, primals_1668, primals_1669, buf1088, 10240, grid=grid(10240), stream=stream0)
        del primals_1669
        # Topologically Sorted Source Nodes: [features_98_conv_0], Original ATen: [aten.convolution]
        buf1089 = extern_kernels.convolution(buf1088, primals_1670, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1089, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1090 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf1091 = buf1090; del buf1090  # reuse
        # Topologically Sorted Source Nodes: [features_98_conv_1, sigmoid_559, mul_330], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf1091, buf1089, primals_1671, primals_1672, primals_1673, primals_1674, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_98_conv_3], Original ATen: [aten.convolution]
        buf1092 = extern_kernels.convolution(buf1091, primals_1675, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf1092, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1093 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_98_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf1092, primals_1676, primals_1677, primals_1678, primals_1679, buf1093, 61440, grid=grid(61440), stream=stream0)
        buf1094 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_561, mul_331, features_98_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf1093, buf1094, 15360, grid=grid(15360), stream=stream0)
        buf1095 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_98_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1681, reinterpret_tensor(buf1094, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1680, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf1095)
        del primals_1681
        buf1096 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_563, mul_332], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf1095, buf1096, 640, grid=grid(640), stream=stream0)
        buf1097 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_98_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1683, buf1096, reinterpret_tensor(primals_1682, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf1097)
        del primals_1683
        buf1098 = buf1093; del buf1093  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_561, mul_331, mul_333], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf1098, buf1097, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_98_conv_7], Original ATen: [aten.convolution]
        buf1099 = extern_kernels.convolution(buf1098, primals_1684, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1099, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1100 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_98_conv_8, add_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf1088, buf1099, primals_1685, primals_1686, primals_1687, primals_1688, buf1100, 10240, grid=grid(10240), stream=stream0)
        del primals_1688
        # Topologically Sorted Source Nodes: [features_99_conv_0], Original ATen: [aten.convolution]
        buf1101 = extern_kernels.convolution(buf1100, primals_1689, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1101, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1102 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf1103 = buf1102; del buf1102  # reuse
        # Topologically Sorted Source Nodes: [features_99_conv_1, sigmoid_566, mul_334], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf1103, buf1101, primals_1690, primals_1691, primals_1692, primals_1693, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_99_conv_3], Original ATen: [aten.convolution]
        buf1104 = extern_kernels.convolution(buf1103, primals_1694, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf1104, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1105 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_99_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf1104, primals_1695, primals_1696, primals_1697, primals_1698, buf1105, 61440, grid=grid(61440), stream=stream0)
        buf1106 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_568, mul_335, features_99_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf1105, buf1106, 15360, grid=grid(15360), stream=stream0)
        buf1107 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_99_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1700, reinterpret_tensor(buf1106, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1699, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf1107)
        del primals_1700
        buf1108 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_570, mul_336], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf1107, buf1108, 640, grid=grid(640), stream=stream0)
        buf1109 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_99_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1702, buf1108, reinterpret_tensor(primals_1701, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf1109)
        del primals_1702
        buf1110 = buf1105; del buf1105  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_568, mul_335, mul_337], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf1110, buf1109, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_99_conv_7], Original ATen: [aten.convolution]
        buf1111 = extern_kernels.convolution(buf1110, primals_1703, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1111, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1112 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_99_conv_8, add_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf1100, buf1111, primals_1704, primals_1705, primals_1706, primals_1707, buf1112, 10240, grid=grid(10240), stream=stream0)
        del primals_1707
        # Topologically Sorted Source Nodes: [features_100_conv_0], Original ATen: [aten.convolution]
        buf1113 = extern_kernels.convolution(buf1112, primals_1708, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1113, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1114 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        buf1115 = buf1114; del buf1114  # reuse
        # Topologically Sorted Source Nodes: [features_100_conv_1, sigmoid_573, mul_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mul_sigmoid_52.run(buf1115, buf1113, primals_1709, primals_1710, primals_1711, primals_1712, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_100_conv_3], Original ATen: [aten.convolution]
        buf1116 = extern_kernels.convolution(buf1115, primals_1713, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=3840, bias=None)
        assert_size_stride(buf1116, (4, 3840, 2, 2), (15360, 1, 7680, 3840))
        buf1117 = empty_strided_cuda((4, 3840, 2, 2), (15360, 1, 7680, 3840), torch.float32)
        # Topologically Sorted Source Nodes: [features_100_conv_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_53.run(buf1116, primals_1714, primals_1715, primals_1716, primals_1717, buf1117, 61440, grid=grid(61440), stream=stream0)
        buf1118 = empty_strided_cuda((4, 3840, 1, 1), (3840, 1, 15360, 15360), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_575, mul_339, features_100_conv_6_avg_pool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_54.run(buf1117, buf1118, 15360, grid=grid(15360), stream=stream0)
        buf1119 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_100_conv_6_fc_0], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1719, reinterpret_tensor(buf1118, (4, 3840), (3840, 1), 0), reinterpret_tensor(primals_1718, (3840, 160), (1, 3840), 0), alpha=1, beta=1, out=buf1119)
        del primals_1719
        buf1120 = empty_strided_cuda((4, 160), (160, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_577, mul_340], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_55.run(buf1119, buf1120, 640, grid=grid(640), stream=stream0)
        buf1121 = empty_strided_cuda((4, 3840), (3840, 1), torch.float32)
        # Topologically Sorted Source Nodes: [features_100_conv_6_fc_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1721, buf1120, reinterpret_tensor(primals_1720, (160, 3840), (1, 160), 0), alpha=1, beta=1, out=buf1121)
        del primals_1721
        buf1122 = buf1117; del buf1117  # reuse
        # Topologically Sorted Source Nodes: [sigmoid_575, mul_339, mul_341], Original ATen: [aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sigmoid_56.run(buf1122, buf1121, 61440, grid=grid(61440), stream=stream0)
        # Topologically Sorted Source Nodes: [features_100_conv_7], Original ATen: [aten.convolution]
        buf1123 = extern_kernels.convolution(buf1122, primals_1722, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1123, (4, 640, 2, 2), (2560, 1, 1280, 640))
        buf1124 = empty_strided_cuda((4, 640, 2, 2), (2560, 1, 1280, 640), torch.float32)
        # Topologically Sorted Source Nodes: [features_100_conv_8, add_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_57.run(buf1112, buf1123, primals_1723, primals_1724, primals_1725, primals_1726, buf1124, 10240, grid=grid(10240), stream=stream0)
        del primals_1726
        # Topologically Sorted Source Nodes: [conv_0], Original ATen: [aten.convolution]
        buf1125 = extern_kernels.convolution(buf1124, primals_1727, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1125, (4, 1792, 2, 2), (7168, 1, 3584, 1792))
        buf1126 = empty_strided_cuda((4, 1792, 2, 2), (7168, 1, 3584, 1792), torch.float32)
        # Topologically Sorted Source Nodes: [conv_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_58.run(buf1125, primals_1728, primals_1729, primals_1730, primals_1731, buf1126, 28672, grid=grid(28672), stream=stream0)
        buf1127 = empty_strided_cuda((4, 1792, 1, 1), (1792, 1, 7168, 7168), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_580, mul_342, avgpool], Original ATen: [aten.sigmoid, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_mul_sigmoid_59.run(buf1126, buf1127, 7168, grid=grid(7168), stream=stream0)
        del buf1126
        buf1128 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [classifier], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_1733, reinterpret_tensor(buf1127, (4, 1792), (1792, 1), 0), reinterpret_tensor(primals_1732, (1792, 1000), (1, 1792), 0), alpha=1, beta=1, out=buf1128)
        del primals_1733
    return (buf1128, buf0, buf1, primals_3, primals_4, primals_5, primals_6, buf2, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, buf3, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, buf4, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, buf5, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, buf6, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, buf7, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, buf8, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, buf9, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, buf10, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, buf11, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, buf12, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, buf13, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, buf14, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, buf15, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, buf16, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, buf17, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, buf18, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, buf19, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, buf20, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, buf21, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_221, primals_222, primals_223, primals_224, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_240, primals_241, primals_242, primals_243, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_259, primals_260, primals_261, primals_262, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_278, primals_279, primals_280, primals_281, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_316, primals_317, primals_318, primals_319, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_335, primals_336, primals_337, primals_338, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_354, primals_355, primals_356, primals_357, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_373, primals_374, primals_375, primals_376, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_411, primals_412, primals_413, primals_414, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_430, primals_431, primals_432, primals_433, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_449, primals_450, primals_451, primals_452, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_468, primals_469, primals_470, primals_471, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_487, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_506, primals_507, primals_508, primals_509, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_525, primals_526, primals_527, primals_528, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_544, primals_545, primals_546, primals_547, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_563, primals_564, primals_565, primals_566, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_582, primals_583, primals_584, primals_585, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_601, primals_602, primals_603, primals_604, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_620, primals_621, primals_622, primals_623, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_639, primals_640, primals_641, primals_642, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_658, primals_659, primals_660, primals_661, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_677, primals_678, primals_679, primals_680, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_696, primals_697, primals_698, primals_699, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_715, primals_716, primals_717, primals_718, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_734, primals_735, primals_736, primals_737, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_753, primals_754, primals_755, primals_756, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_772, primals_773, primals_774, primals_775, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_791, primals_792, primals_793, primals_794, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_810, primals_811, primals_812, primals_813, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_829, primals_830, primals_831, primals_832, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_848, primals_849, primals_850, primals_851, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_867, primals_868, primals_869, primals_870, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_886, primals_887, primals_888, primals_889, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_905, primals_906, primals_907, primals_908, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_924, primals_925, primals_926, primals_927, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_943, primals_944, primals_945, primals_946, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_962, primals_963, primals_964, primals_965, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_981, primals_982, primals_983, primals_984, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_1000, primals_1001, primals_1002, primals_1003, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1019, primals_1020, primals_1021, primals_1022, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1038, primals_1039, primals_1040, primals_1041, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1057, primals_1058, primals_1059, primals_1060, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1076, primals_1077, primals_1078, primals_1079, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1095, primals_1096, primals_1097, primals_1098, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1114, primals_1115, primals_1116, primals_1117, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1133, primals_1134, primals_1135, primals_1136, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1152, primals_1153, primals_1154, primals_1155, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1171, primals_1172, primals_1173, primals_1174, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1190, primals_1191, primals_1192, primals_1193, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1209, primals_1210, primals_1211, primals_1212, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1228, primals_1229, primals_1230, primals_1231, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1247, primals_1248, primals_1249, primals_1250, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1266, primals_1267, primals_1268, primals_1269, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1285, primals_1286, primals_1287, primals_1288, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1304, primals_1305, primals_1306, primals_1307, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1323, primals_1324, primals_1325, primals_1326, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1342, primals_1343, primals_1344, primals_1345, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352, primals_1353, primals_1354, primals_1355, primals_1356, primals_1361, primals_1362, primals_1363, primals_1364, primals_1366, primals_1367, primals_1368, primals_1369, primals_1370, primals_1371, primals_1372, primals_1373, primals_1374, primals_1375, primals_1380, primals_1381, primals_1382, primals_1383, primals_1385, primals_1386, primals_1387, primals_1388, primals_1389, primals_1390, primals_1391, primals_1392, primals_1393, primals_1394, primals_1399, primals_1400, primals_1401, primals_1402, primals_1404, primals_1405, primals_1406, primals_1407, primals_1408, primals_1409, primals_1410, primals_1411, primals_1412, primals_1413, primals_1418, primals_1419, primals_1420, primals_1421, primals_1423, primals_1424, primals_1425, primals_1426, primals_1427, primals_1428, primals_1429, primals_1430, primals_1431, primals_1432, primals_1437, primals_1438, primals_1439, primals_1440, primals_1442, primals_1443, primals_1444, primals_1445, primals_1446, primals_1447, primals_1448, primals_1449, primals_1450, primals_1451, primals_1456, primals_1457, primals_1458, primals_1459, primals_1461, primals_1462, primals_1463, primals_1464, primals_1465, primals_1466, primals_1467, primals_1468, primals_1469, primals_1470, primals_1475, primals_1476, primals_1477, primals_1478, primals_1480, primals_1481, primals_1482, primals_1483, primals_1484, primals_1485, primals_1486, primals_1487, primals_1488, primals_1489, primals_1494, primals_1495, primals_1496, primals_1497, primals_1499, primals_1500, primals_1501, primals_1502, primals_1503, primals_1504, primals_1505, primals_1506, primals_1507, primals_1508, primals_1513, primals_1514, primals_1515, primals_1516, primals_1518, primals_1519, primals_1520, primals_1521, primals_1522, primals_1523, primals_1524, primals_1525, primals_1526, primals_1527, primals_1532, primals_1533, primals_1534, primals_1535, primals_1537, primals_1538, primals_1539, primals_1540, primals_1541, primals_1542, primals_1543, primals_1544, primals_1545, primals_1546, primals_1551, primals_1552, primals_1553, primals_1554, primals_1556, primals_1557, primals_1558, primals_1559, primals_1560, primals_1561, primals_1562, primals_1563, primals_1564, primals_1565, primals_1570, primals_1571, primals_1572, primals_1573, primals_1575, primals_1576, primals_1577, primals_1578, primals_1579, primals_1580, primals_1581, primals_1582, primals_1583, primals_1584, primals_1589, primals_1590, primals_1591, primals_1592, primals_1594, primals_1595, primals_1596, primals_1597, primals_1598, primals_1599, primals_1600, primals_1601, primals_1602, primals_1603, primals_1608, primals_1609, primals_1610, primals_1611, primals_1613, primals_1614, primals_1615, primals_1616, primals_1617, primals_1618, primals_1619, primals_1620, primals_1621, primals_1622, primals_1627, primals_1628, primals_1629, primals_1630, primals_1632, primals_1633, primals_1634, primals_1635, primals_1636, primals_1637, primals_1638, primals_1639, primals_1640, primals_1641, primals_1646, primals_1647, primals_1648, primals_1649, primals_1651, primals_1652, primals_1653, primals_1654, primals_1655, primals_1656, primals_1657, primals_1658, primals_1659, primals_1660, primals_1665, primals_1666, primals_1667, primals_1668, primals_1670, primals_1671, primals_1672, primals_1673, primals_1674, primals_1675, primals_1676, primals_1677, primals_1678, primals_1679, primals_1684, primals_1685, primals_1686, primals_1687, primals_1689, primals_1690, primals_1691, primals_1692, primals_1693, primals_1694, primals_1695, primals_1696, primals_1697, primals_1698, primals_1703, primals_1704, primals_1705, primals_1706, primals_1708, primals_1709, primals_1710, primals_1711, primals_1712, primals_1713, primals_1714, primals_1715, primals_1716, primals_1717, primals_1722, primals_1723, primals_1724, primals_1725, primals_1727, primals_1728, primals_1729, primals_1730, primals_1731, buf22, buf24, buf25, buf27, buf28, buf29, buf30, buf32, buf33, buf34, buf35, buf37, buf38, buf39, buf40, buf42, buf43, buf44, buf45, buf47, buf48, buf49, buf50, buf52, buf53, buf54, buf55, buf57, buf58, buf59, buf60, buf62, buf63, buf64, buf65, buf67, buf68, buf69, buf70, buf72, buf73, buf74, buf75, buf77, buf78, buf79, buf80, buf82, buf83, buf84, buf85, buf87, buf88, buf89, buf90, buf92, buf93, buf94, buf95, buf97, buf98, buf99, buf100, buf102, buf103, buf104, buf105, buf107, buf108, buf109, buf110, buf112, buf113, buf114, buf115, buf117, buf118, buf119, buf120, buf122, buf123, buf124, buf125, buf127, buf128, reinterpret_tensor(buf131, (4, 384), (384, 1), 0), buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf140, buf141, reinterpret_tensor(buf144, (4, 768), (768, 1), 0), buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf153, buf154, reinterpret_tensor(buf157, (4, 768), (768, 1), 0), buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf166, buf167, reinterpret_tensor(buf170, (4, 768), (768, 1), 0), buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf179, buf180, reinterpret_tensor(buf183, (4, 768), (768, 1), 0), buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf192, buf193, reinterpret_tensor(buf196, (4, 768), (768, 1), 0), buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf205, buf206, reinterpret_tensor(buf209, (4, 768), (768, 1), 0), buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf218, buf219, reinterpret_tensor(buf222, (4, 768), (768, 1), 0), buf223, buf224, buf225, buf226, buf227, buf228, buf229, buf231, buf232, reinterpret_tensor(buf235, (4, 768), (768, 1), 0), buf236, buf237, buf238, buf239, buf240, buf241, buf242, buf244, buf245, reinterpret_tensor(buf248, (4, 768), (768, 1), 0), buf249, buf250, buf251, buf252, buf253, buf254, buf255, buf257, buf258, reinterpret_tensor(buf261, (4, 768), (768, 1), 0), buf262, buf263, buf264, buf265, buf266, buf267, buf268, buf270, buf271, reinterpret_tensor(buf274, (4, 768), (768, 1), 0), buf275, buf276, buf277, buf278, buf279, buf280, buf281, buf283, buf284, reinterpret_tensor(buf287, (4, 768), (768, 1), 0), buf288, buf289, buf290, buf291, buf292, buf293, buf294, buf296, buf297, reinterpret_tensor(buf300, (4, 768), (768, 1), 0), buf301, buf302, buf303, buf304, buf305, buf306, buf307, buf309, buf310, reinterpret_tensor(buf313, (4, 768), (768, 1), 0), buf314, buf315, buf316, buf317, buf318, buf319, buf320, buf322, buf323, reinterpret_tensor(buf326, (4, 768), (768, 1), 0), buf327, buf328, buf329, buf330, buf331, buf332, buf333, buf335, buf336, reinterpret_tensor(buf339, (4, 1152), (1152, 1), 0), buf340, buf341, buf342, buf343, buf344, buf345, buf346, buf348, buf349, reinterpret_tensor(buf352, (4, 1536), (1536, 1), 0), buf353, buf354, buf355, buf356, buf357, buf358, buf359, buf361, buf362, reinterpret_tensor(buf365, (4, 1536), (1536, 1), 0), buf366, buf367, buf368, buf369, buf370, buf371, buf372, buf374, buf375, reinterpret_tensor(buf378, (4, 1536), (1536, 1), 0), buf379, buf380, buf381, buf382, buf383, buf384, buf385, buf387, buf388, reinterpret_tensor(buf391, (4, 1536), (1536, 1), 0), buf392, buf393, buf394, buf395, buf396, buf397, buf398, buf400, buf401, reinterpret_tensor(buf404, (4, 1536), (1536, 1), 0), buf405, buf406, buf407, buf408, buf409, buf410, buf411, buf413, buf414, reinterpret_tensor(buf417, (4, 1536), (1536, 1), 0), buf418, buf419, buf420, buf421, buf422, buf423, buf424, buf426, buf427, reinterpret_tensor(buf430, (4, 1536), (1536, 1), 0), buf431, buf432, buf433, buf434, buf435, buf436, buf437, buf439, buf440, reinterpret_tensor(buf443, (4, 1536), (1536, 1), 0), buf444, buf445, buf446, buf447, buf448, buf449, buf450, buf452, buf453, reinterpret_tensor(buf456, (4, 1536), (1536, 1), 0), buf457, buf458, buf459, buf460, buf461, buf462, buf463, buf465, buf466, reinterpret_tensor(buf469, (4, 1536), (1536, 1), 0), buf470, buf471, buf472, buf473, buf474, buf475, buf476, buf478, buf479, reinterpret_tensor(buf482, (4, 1536), (1536, 1), 0), buf483, buf484, buf485, buf486, buf487, buf488, buf489, buf491, buf492, reinterpret_tensor(buf495, (4, 1536), (1536, 1), 0), buf496, buf497, buf498, buf499, buf500, buf501, buf502, buf504, buf505, reinterpret_tensor(buf508, (4, 1536), (1536, 1), 0), buf509, buf510, buf511, buf512, buf513, buf514, buf515, buf517, buf518, reinterpret_tensor(buf521, (4, 1536), (1536, 1), 0), buf522, buf523, buf524, buf525, buf526, buf527, buf528, buf530, buf531, reinterpret_tensor(buf534, (4, 1536), (1536, 1), 0), buf535, buf536, buf537, buf538, buf539, buf540, buf541, buf543, buf544, reinterpret_tensor(buf547, (4, 1536), (1536, 1), 0), buf548, buf549, buf550, buf551, buf552, buf553, buf554, buf556, buf557, reinterpret_tensor(buf560, (4, 1536), (1536, 1), 0), buf561, buf562, buf563, buf564, buf565, buf566, buf567, buf569, buf570, reinterpret_tensor(buf573, (4, 1536), (1536, 1), 0), buf574, buf575, buf576, buf577, buf578, buf579, buf580, buf582, buf583, reinterpret_tensor(buf586, (4, 1536), (1536, 1), 0), buf587, buf588, buf589, buf590, buf591, buf592, buf593, buf595, buf596, reinterpret_tensor(buf599, (4, 1536), (1536, 1), 0), buf600, buf601, buf602, buf603, buf604, buf605, buf606, buf608, buf609, reinterpret_tensor(buf612, (4, 1536), (1536, 1), 0), buf613, buf614, buf615, buf616, buf617, buf618, buf619, buf621, buf622, reinterpret_tensor(buf625, (4, 1536), (1536, 1), 0), buf626, buf627, buf628, buf629, buf630, buf631, buf632, buf634, buf635, reinterpret_tensor(buf638, (4, 1536), (1536, 1), 0), buf639, buf640, buf641, buf642, buf643, buf644, buf645, buf647, buf648, reinterpret_tensor(buf650, (4, 1536), (1536, 1), 0), buf651, buf652, buf653, buf654, buf655, buf656, buf657, buf659, buf660, reinterpret_tensor(buf662, (4, 3072), (3072, 1), 0), buf663, buf664, buf665, buf666, buf667, buf668, buf669, buf671, buf672, reinterpret_tensor(buf674, (4, 3072), (3072, 1), 0), buf675, buf676, buf677, buf678, buf679, buf680, buf681, buf683, buf684, reinterpret_tensor(buf686, (4, 3072), (3072, 1), 0), buf687, buf688, buf689, buf690, buf691, buf692, buf693, buf695, buf696, reinterpret_tensor(buf698, (4, 3072), (3072, 1), 0), buf699, buf700, buf701, buf702, buf703, buf704, buf705, buf707, buf708, reinterpret_tensor(buf710, (4, 3072), (3072, 1), 0), buf711, buf712, buf713, buf714, buf715, buf716, buf717, buf719, buf720, reinterpret_tensor(buf722, (4, 3072), (3072, 1), 0), buf723, buf724, buf725, buf726, buf727, buf728, buf729, buf731, buf732, reinterpret_tensor(buf734, (4, 3072), (3072, 1), 0), buf735, buf736, buf737, buf738, buf739, buf740, buf741, buf743, buf744, reinterpret_tensor(buf746, (4, 3072), (3072, 1), 0), buf747, buf748, buf749, buf750, buf751, buf752, buf753, buf755, buf756, reinterpret_tensor(buf758, (4, 3072), (3072, 1), 0), buf759, buf760, buf761, buf762, buf763, buf764, buf765, buf767, buf768, reinterpret_tensor(buf770, (4, 3072), (3072, 1), 0), buf771, buf772, buf773, buf774, buf775, buf776, buf777, buf779, buf780, reinterpret_tensor(buf782, (4, 3072), (3072, 1), 0), buf783, buf784, buf785, buf786, buf787, buf788, buf789, buf791, buf792, reinterpret_tensor(buf794, (4, 3072), (3072, 1), 0), buf795, buf796, buf797, buf798, buf799, buf800, buf801, buf803, buf804, reinterpret_tensor(buf806, (4, 3072), (3072, 1), 0), buf807, buf808, buf809, buf810, buf811, buf812, buf813, buf815, buf816, reinterpret_tensor(buf818, (4, 3072), (3072, 1), 0), buf819, buf820, buf821, buf822, buf823, buf824, buf825, buf827, buf828, reinterpret_tensor(buf830, (4, 3072), (3072, 1), 0), buf831, buf832, buf833, buf834, buf835, buf836, buf837, buf839, buf840, reinterpret_tensor(buf842, (4, 3072), (3072, 1), 0), buf843, buf844, buf845, buf846, buf847, buf848, buf849, buf851, buf852, reinterpret_tensor(buf854, (4, 3072), (3072, 1), 0), buf855, buf856, buf857, buf858, buf859, buf860, buf861, buf863, buf864, reinterpret_tensor(buf866, (4, 3072), (3072, 1), 0), buf867, buf868, buf869, buf870, buf871, buf872, buf873, buf875, buf876, reinterpret_tensor(buf878, (4, 3072), (3072, 1), 0), buf879, buf880, buf881, buf882, buf883, buf884, buf885, buf887, buf888, reinterpret_tensor(buf890, (4, 3072), (3072, 1), 0), buf891, buf892, buf893, buf894, buf895, buf896, buf897, buf899, buf900, reinterpret_tensor(buf902, (4, 3072), (3072, 1), 0), buf903, buf904, buf905, buf906, buf907, buf908, buf909, buf911, buf912, reinterpret_tensor(buf914, (4, 3072), (3072, 1), 0), buf915, buf916, buf917, buf918, buf919, buf920, buf921, buf923, buf924, reinterpret_tensor(buf926, (4, 3072), (3072, 1), 0), buf927, buf928, buf929, buf930, buf931, buf932, buf933, buf935, buf936, reinterpret_tensor(buf938, (4, 3072), (3072, 1), 0), buf939, buf940, buf941, buf942, buf943, buf944, buf945, buf947, buf948, reinterpret_tensor(buf950, (4, 3072), (3072, 1), 0), buf951, buf952, buf953, buf954, buf955, buf956, buf957, buf959, buf960, reinterpret_tensor(buf962, (4, 3072), (3072, 1), 0), buf963, buf964, buf965, buf966, buf967, buf968, buf969, buf971, buf972, reinterpret_tensor(buf974, (4, 3072), (3072, 1), 0), buf975, buf976, buf977, buf978, buf979, buf980, buf981, buf983, buf984, reinterpret_tensor(buf986, (4, 3072), (3072, 1), 0), buf987, buf988, buf989, buf990, buf991, buf992, buf993, buf995, buf996, reinterpret_tensor(buf998, (4, 3072), (3072, 1), 0), buf999, buf1000, buf1001, buf1002, buf1003, buf1004, buf1005, buf1007, buf1008, reinterpret_tensor(buf1010, (4, 3072), (3072, 1), 0), buf1011, buf1012, buf1013, buf1014, buf1015, buf1016, buf1017, buf1019, buf1020, reinterpret_tensor(buf1022, (4, 3072), (3072, 1), 0), buf1023, buf1024, buf1025, buf1026, buf1027, buf1028, buf1029, buf1031, buf1032, reinterpret_tensor(buf1034, (4, 3072), (3072, 1), 0), buf1035, buf1036, buf1037, buf1038, buf1039, buf1040, buf1041, buf1043, buf1044, reinterpret_tensor(buf1046, (4, 3840), (3840, 1), 0), buf1047, buf1048, buf1049, buf1050, buf1051, buf1052, buf1053, buf1055, buf1056, reinterpret_tensor(buf1058, (4, 3840), (3840, 1), 0), buf1059, buf1060, buf1061, buf1062, buf1063, buf1064, buf1065, buf1067, buf1068, reinterpret_tensor(buf1070, (4, 3840), (3840, 1), 0), buf1071, buf1072, buf1073, buf1074, buf1075, buf1076, buf1077, buf1079, buf1080, reinterpret_tensor(buf1082, (4, 3840), (3840, 1), 0), buf1083, buf1084, buf1085, buf1086, buf1087, buf1088, buf1089, buf1091, buf1092, reinterpret_tensor(buf1094, (4, 3840), (3840, 1), 0), buf1095, buf1096, buf1097, buf1098, buf1099, buf1100, buf1101, buf1103, buf1104, reinterpret_tensor(buf1106, (4, 3840), (3840, 1), 0), buf1107, buf1108, buf1109, buf1110, buf1111, buf1112, buf1113, buf1115, buf1116, reinterpret_tensor(buf1118, (4, 3840), (3840, 1), 0), buf1119, buf1120, buf1121, buf1122, buf1123, buf1124, buf1125, reinterpret_tensor(buf1127, (4, 1792), (1792, 1), 0), primals_1732, primals_1720, primals_1718, primals_1701, primals_1699, primals_1682, primals_1680, primals_1663, primals_1661, primals_1644, primals_1642, primals_1625, primals_1623, primals_1606, primals_1604, primals_1587, primals_1585, primals_1568, primals_1566, primals_1549, primals_1547, primals_1530, primals_1528, primals_1511, primals_1509, primals_1492, primals_1490, primals_1473, primals_1471, primals_1454, primals_1452, primals_1435, primals_1433, primals_1416, primals_1414, primals_1397, primals_1395, primals_1378, primals_1376, primals_1359, primals_1357, primals_1340, primals_1338, primals_1321, primals_1319, primals_1302, primals_1300, primals_1283, primals_1281, primals_1264, primals_1262, primals_1245, primals_1243, primals_1226, primals_1224, primals_1207, primals_1205, primals_1188, primals_1186, primals_1169, primals_1167, primals_1150, primals_1148, primals_1131, primals_1129, primals_1112, primals_1110, primals_1093, primals_1091, primals_1074, primals_1072, primals_1055, primals_1053, primals_1036, primals_1034, primals_1017, primals_1015, primals_998, primals_996, primals_979, primals_977, primals_960, primals_958, primals_941, primals_939, primals_922, primals_920, primals_903, primals_901, primals_884, primals_882, primals_865, primals_863, primals_846, primals_844, primals_827, primals_825, primals_808, primals_806, primals_789, primals_787, primals_770, primals_768, primals_751, primals_749, primals_732, primals_730, primals_713, primals_711, primals_694, primals_692, primals_675, primals_673, primals_656, primals_654, primals_637, primals_635, primals_618, primals_616, primals_599, primals_597, primals_580, primals_578, primals_561, primals_559, primals_542, primals_540, primals_523, primals_521, primals_504, primals_502, primals_485, primals_483, primals_466, primals_464, primals_447, primals_445, primals_428, primals_426, primals_409, primals_407, primals_390, primals_388, primals_371, primals_369, primals_352, primals_350, primals_333, primals_331, primals_314, primals_312, primals_295, primals_293, primals_276, primals_274, primals_257, primals_255, primals_238, primals_236, primals_219, primals_217, )


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
    primals_122 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    primals_187 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((96, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((24, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((384, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((768, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((48, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((768, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((48, 1152), (1152, 1), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((1152, 48), (48, 1), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((256, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_876 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_879 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_882 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_885 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_888 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_891 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_894 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_897 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_900 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_903 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_906 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_909 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_912 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_915 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_918 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_921 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_924 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_927 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_930 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_933 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_936 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_937 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_938 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_939 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_940 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_941 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_942 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_943 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_944 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_945 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_946 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_947 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_948 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_949 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_950 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_951 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_952 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_953 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_954 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_955 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_956 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_957 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_958 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_959 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_960 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_961 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_962 = rand_strided((256, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_963 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_964 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_965 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_966 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_967 = rand_strided((1536, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_968 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_969 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_970 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_971 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_972 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_973 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_974 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_975 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_976 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_977 = rand_strided((64, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_978 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_979 = rand_strided((1536, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_980 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_981 = rand_strided((512, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_982 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_983 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_984 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_985 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_986 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_987 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_988 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_989 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_990 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_991 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_992 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_993 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_994 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_995 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_996 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_997 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_998 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_999 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1000 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1001 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1002 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1003 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1004 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1005 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1006 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1007 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1008 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1009 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1010 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1011 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1012 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1013 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1014 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1015 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1016 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1017 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1018 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1019 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1020 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1021 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1022 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1023 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1024 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1025 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1026 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1027 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1028 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1029 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1030 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1031 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1032 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1033 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1034 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1035 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1036 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1037 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1038 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1039 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1040 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1041 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1042 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1043 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1044 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1045 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1046 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1047 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1048 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1049 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1050 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1051 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1052 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1053 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1054 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1055 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1056 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1057 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1058 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1059 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1060 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1061 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1062 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1063 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1064 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1065 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1066 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1067 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1068 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1069 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1070 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1071 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1072 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1073 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1074 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1075 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1076 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1077 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1078 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1079 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1080 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1081 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1082 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1083 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1084 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1085 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1086 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1087 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1088 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1089 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1090 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1091 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1092 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1093 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1094 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1095 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1096 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1097 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1098 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1099 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1100 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1101 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1102 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1103 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1104 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1105 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1106 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1107 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1108 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1109 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1110 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1111 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1112 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1113 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1114 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1118 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1119 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1120 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1121 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1122 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1123 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1124 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1125 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1126 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1127 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1128 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1129 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1131 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1132 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1133 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1136 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1137 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1138 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1139 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1140 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1141 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1142 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1143 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1144 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1145 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1146 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1147 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1148 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1149 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1150 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1151 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1152 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1153 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1154 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1155 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1156 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1157 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1158 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1159 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1160 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1161 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1162 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1163 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1164 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1165 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1166 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1167 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1169 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1170 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1171 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1172 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1175 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1176 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1177 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1178 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1179 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1180 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1181 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1182 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1183 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1184 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1185 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1186 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1188 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1189 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1190 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1192 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1193 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1195 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1196 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1197 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1198 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1199 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1200 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1201 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1202 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1203 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1204 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1205 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1206 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1207 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1208 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1209 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1211 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1212 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1214 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1215 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1216 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1217 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1218 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1219 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1220 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1221 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1222 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1223 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1224 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1225 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1226 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1227 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1228 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1229 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1230 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1233 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1234 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1235 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1236 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1237 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1238 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1239 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1240 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1241 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1242 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1243 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1245 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1246 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1247 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1249 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1251 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1252 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1253 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1254 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1255 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1256 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1257 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1258 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1259 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1260 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1261 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1262 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1264 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1265 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1266 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1267 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1269 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1271 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1272 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1273 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1274 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1275 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1276 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1277 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1278 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1279 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1280 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1281 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1283 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1284 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1285 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1287 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1288 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1289 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1290 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1291 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1292 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1293 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1294 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1295 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1296 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1297 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1298 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1299 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1300 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1301 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1302 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1303 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1304 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1306 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1307 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1309 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1310 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1311 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1312 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1313 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1314 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1315 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1316 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1317 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1318 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1319 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1320 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1321 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1322 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1323 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1324 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1325 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1326 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1327 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1328 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1329 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1330 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1331 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1332 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1333 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1334 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1335 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1336 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1337 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1338 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1339 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1340 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1341 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1342 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1343 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1344 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1345 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1346 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1347 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1348 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1349 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1350 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1351 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1352 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1353 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1354 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1355 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1356 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1357 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1358 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1359 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1360 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1361 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1362 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1363 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1364 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1365 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1366 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1367 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1368 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1369 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1370 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1371 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1372 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1373 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1374 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1375 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1376 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1377 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1378 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1379 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1380 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1381 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1382 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1383 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1384 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1385 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1386 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1387 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1388 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1389 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1390 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1391 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1392 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1393 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1394 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1395 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1397 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1398 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1399 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1400 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1401 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1402 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1403 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1404 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1405 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1406 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1407 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1408 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1409 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1410 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1411 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1412 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1413 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1414 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1416 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1417 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1418 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1419 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1422 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1423 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1424 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1425 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1426 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1427 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1428 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1429 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1430 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1431 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1432 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1433 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1434 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1435 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1436 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1437 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1438 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1439 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1440 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1441 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1442 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1443 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1444 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1445 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1446 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1447 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1448 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1449 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1450 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1451 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1452 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1454 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1455 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1456 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1457 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1458 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1459 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1460 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1461 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1462 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1463 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1464 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1465 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1466 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1467 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1468 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1469 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1470 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1471 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1472 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1473 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1474 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1475 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1476 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1477 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1478 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1479 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1480 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1481 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1482 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1483 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1484 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1485 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1486 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1487 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1488 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1489 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1490 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1491 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1492 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1493 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1494 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1495 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1496 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1497 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1498 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1499 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1500 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1501 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1502 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1503 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1504 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1505 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1506 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1507 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1508 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1509 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1510 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1511 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1512 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1513 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1514 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1515 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1516 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1517 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1518 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1519 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1520 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1521 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1522 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1523 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1524 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1525 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1526 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1527 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1528 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1529 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1530 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1531 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1532 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1533 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1534 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1535 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1536 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1537 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1538 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1539 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1540 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1541 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1542 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1543 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1544 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1545 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1546 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1547 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1548 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1549 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1550 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1551 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1552 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1553 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1554 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1555 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1556 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1557 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1558 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1559 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1560 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1561 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1562 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1563 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1564 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1565 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1566 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1567 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1568 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1569 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1570 = rand_strided((512, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1571 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1572 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1573 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1574 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1575 = rand_strided((3072, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1576 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1577 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1578 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1579 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1580 = rand_strided((3072, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1581 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1582 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1583 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1584 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1585 = rand_strided((128, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_1586 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1587 = rand_strided((3072, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_1588 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1589 = rand_strided((640, 3072, 1, 1), (3072, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1590 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1591 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1592 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1593 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1594 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1595 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1596 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1597 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1598 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1599 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1600 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1601 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1602 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1603 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1604 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1605 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1606 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1607 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1608 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1609 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1610 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1611 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1612 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1613 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1614 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1615 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1616 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1617 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1618 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1619 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1620 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1621 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1622 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1623 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1624 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1625 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1626 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1627 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1628 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1629 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1630 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1631 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1632 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1633 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1634 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1635 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1636 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1637 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1638 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1639 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1640 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1641 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1642 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1643 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1644 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1645 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1646 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1647 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1648 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1649 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1650 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1651 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1652 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1653 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1654 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1655 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1656 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1657 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1658 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1659 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1660 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1661 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1662 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1663 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1664 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1665 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1666 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1667 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1668 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1669 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1670 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1671 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1672 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1673 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1674 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1675 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1676 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1677 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1678 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1679 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1680 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1681 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1682 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1683 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1684 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1685 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1686 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1687 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1688 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1689 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1690 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1691 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1692 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1693 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1694 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1695 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1696 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1697 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1698 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1699 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1700 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1701 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1702 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1703 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1704 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1705 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1706 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1707 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1708 = rand_strided((3840, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1709 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1710 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1711 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1712 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1713 = rand_strided((3840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_1714 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1715 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1716 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1717 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1718 = rand_strided((160, 3840), (3840, 1), device='cuda:0', dtype=torch.float32)
    primals_1719 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1720 = rand_strided((3840, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    primals_1721 = rand_strided((3840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1722 = rand_strided((640, 3840, 1, 1), (3840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1723 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1724 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1725 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1726 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1727 = rand_strided((1792, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_1728 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1729 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1730 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1731 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_1732 = rand_strided((1000, 1792), (1792, 1), device='cuda:0', dtype=torch.float32)
    primals_1733 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936, primals_937, primals_938, primals_939, primals_940, primals_941, primals_942, primals_943, primals_944, primals_945, primals_946, primals_947, primals_948, primals_949, primals_950, primals_951, primals_952, primals_953, primals_954, primals_955, primals_956, primals_957, primals_958, primals_959, primals_960, primals_961, primals_962, primals_963, primals_964, primals_965, primals_966, primals_967, primals_968, primals_969, primals_970, primals_971, primals_972, primals_973, primals_974, primals_975, primals_976, primals_977, primals_978, primals_979, primals_980, primals_981, primals_982, primals_983, primals_984, primals_985, primals_986, primals_987, primals_988, primals_989, primals_990, primals_991, primals_992, primals_993, primals_994, primals_995, primals_996, primals_997, primals_998, primals_999, primals_1000, primals_1001, primals_1002, primals_1003, primals_1004, primals_1005, primals_1006, primals_1007, primals_1008, primals_1009, primals_1010, primals_1011, primals_1012, primals_1013, primals_1014, primals_1015, primals_1016, primals_1017, primals_1018, primals_1019, primals_1020, primals_1021, primals_1022, primals_1023, primals_1024, primals_1025, primals_1026, primals_1027, primals_1028, primals_1029, primals_1030, primals_1031, primals_1032, primals_1033, primals_1034, primals_1035, primals_1036, primals_1037, primals_1038, primals_1039, primals_1040, primals_1041, primals_1042, primals_1043, primals_1044, primals_1045, primals_1046, primals_1047, primals_1048, primals_1049, primals_1050, primals_1051, primals_1052, primals_1053, primals_1054, primals_1055, primals_1056, primals_1057, primals_1058, primals_1059, primals_1060, primals_1061, primals_1062, primals_1063, primals_1064, primals_1065, primals_1066, primals_1067, primals_1068, primals_1069, primals_1070, primals_1071, primals_1072, primals_1073, primals_1074, primals_1075, primals_1076, primals_1077, primals_1078, primals_1079, primals_1080, primals_1081, primals_1082, primals_1083, primals_1084, primals_1085, primals_1086, primals_1087, primals_1088, primals_1089, primals_1090, primals_1091, primals_1092, primals_1093, primals_1094, primals_1095, primals_1096, primals_1097, primals_1098, primals_1099, primals_1100, primals_1101, primals_1102, primals_1103, primals_1104, primals_1105, primals_1106, primals_1107, primals_1108, primals_1109, primals_1110, primals_1111, primals_1112, primals_1113, primals_1114, primals_1115, primals_1116, primals_1117, primals_1118, primals_1119, primals_1120, primals_1121, primals_1122, primals_1123, primals_1124, primals_1125, primals_1126, primals_1127, primals_1128, primals_1129, primals_1130, primals_1131, primals_1132, primals_1133, primals_1134, primals_1135, primals_1136, primals_1137, primals_1138, primals_1139, primals_1140, primals_1141, primals_1142, primals_1143, primals_1144, primals_1145, primals_1146, primals_1147, primals_1148, primals_1149, primals_1150, primals_1151, primals_1152, primals_1153, primals_1154, primals_1155, primals_1156, primals_1157, primals_1158, primals_1159, primals_1160, primals_1161, primals_1162, primals_1163, primals_1164, primals_1165, primals_1166, primals_1167, primals_1168, primals_1169, primals_1170, primals_1171, primals_1172, primals_1173, primals_1174, primals_1175, primals_1176, primals_1177, primals_1178, primals_1179, primals_1180, primals_1181, primals_1182, primals_1183, primals_1184, primals_1185, primals_1186, primals_1187, primals_1188, primals_1189, primals_1190, primals_1191, primals_1192, primals_1193, primals_1194, primals_1195, primals_1196, primals_1197, primals_1198, primals_1199, primals_1200, primals_1201, primals_1202, primals_1203, primals_1204, primals_1205, primals_1206, primals_1207, primals_1208, primals_1209, primals_1210, primals_1211, primals_1212, primals_1213, primals_1214, primals_1215, primals_1216, primals_1217, primals_1218, primals_1219, primals_1220, primals_1221, primals_1222, primals_1223, primals_1224, primals_1225, primals_1226, primals_1227, primals_1228, primals_1229, primals_1230, primals_1231, primals_1232, primals_1233, primals_1234, primals_1235, primals_1236, primals_1237, primals_1238, primals_1239, primals_1240, primals_1241, primals_1242, primals_1243, primals_1244, primals_1245, primals_1246, primals_1247, primals_1248, primals_1249, primals_1250, primals_1251, primals_1252, primals_1253, primals_1254, primals_1255, primals_1256, primals_1257, primals_1258, primals_1259, primals_1260, primals_1261, primals_1262, primals_1263, primals_1264, primals_1265, primals_1266, primals_1267, primals_1268, primals_1269, primals_1270, primals_1271, primals_1272, primals_1273, primals_1274, primals_1275, primals_1276, primals_1277, primals_1278, primals_1279, primals_1280, primals_1281, primals_1282, primals_1283, primals_1284, primals_1285, primals_1286, primals_1287, primals_1288, primals_1289, primals_1290, primals_1291, primals_1292, primals_1293, primals_1294, primals_1295, primals_1296, primals_1297, primals_1298, primals_1299, primals_1300, primals_1301, primals_1302, primals_1303, primals_1304, primals_1305, primals_1306, primals_1307, primals_1308, primals_1309, primals_1310, primals_1311, primals_1312, primals_1313, primals_1314, primals_1315, primals_1316, primals_1317, primals_1318, primals_1319, primals_1320, primals_1321, primals_1322, primals_1323, primals_1324, primals_1325, primals_1326, primals_1327, primals_1328, primals_1329, primals_1330, primals_1331, primals_1332, primals_1333, primals_1334, primals_1335, primals_1336, primals_1337, primals_1338, primals_1339, primals_1340, primals_1341, primals_1342, primals_1343, primals_1344, primals_1345, primals_1346, primals_1347, primals_1348, primals_1349, primals_1350, primals_1351, primals_1352, primals_1353, primals_1354, primals_1355, primals_1356, primals_1357, primals_1358, primals_1359, primals_1360, primals_1361, primals_1362, primals_1363, primals_1364, primals_1365, primals_1366, primals_1367, primals_1368, primals_1369, primals_1370, primals_1371, primals_1372, primals_1373, primals_1374, primals_1375, primals_1376, primals_1377, primals_1378, primals_1379, primals_1380, primals_1381, primals_1382, primals_1383, primals_1384, primals_1385, primals_1386, primals_1387, primals_1388, primals_1389, primals_1390, primals_1391, primals_1392, primals_1393, primals_1394, primals_1395, primals_1396, primals_1397, primals_1398, primals_1399, primals_1400, primals_1401, primals_1402, primals_1403, primals_1404, primals_1405, primals_1406, primals_1407, primals_1408, primals_1409, primals_1410, primals_1411, primals_1412, primals_1413, primals_1414, primals_1415, primals_1416, primals_1417, primals_1418, primals_1419, primals_1420, primals_1421, primals_1422, primals_1423, primals_1424, primals_1425, primals_1426, primals_1427, primals_1428, primals_1429, primals_1430, primals_1431, primals_1432, primals_1433, primals_1434, primals_1435, primals_1436, primals_1437, primals_1438, primals_1439, primals_1440, primals_1441, primals_1442, primals_1443, primals_1444, primals_1445, primals_1446, primals_1447, primals_1448, primals_1449, primals_1450, primals_1451, primals_1452, primals_1453, primals_1454, primals_1455, primals_1456, primals_1457, primals_1458, primals_1459, primals_1460, primals_1461, primals_1462, primals_1463, primals_1464, primals_1465, primals_1466, primals_1467, primals_1468, primals_1469, primals_1470, primals_1471, primals_1472, primals_1473, primals_1474, primals_1475, primals_1476, primals_1477, primals_1478, primals_1479, primals_1480, primals_1481, primals_1482, primals_1483, primals_1484, primals_1485, primals_1486, primals_1487, primals_1488, primals_1489, primals_1490, primals_1491, primals_1492, primals_1493, primals_1494, primals_1495, primals_1496, primals_1497, primals_1498, primals_1499, primals_1500, primals_1501, primals_1502, primals_1503, primals_1504, primals_1505, primals_1506, primals_1507, primals_1508, primals_1509, primals_1510, primals_1511, primals_1512, primals_1513, primals_1514, primals_1515, primals_1516, primals_1517, primals_1518, primals_1519, primals_1520, primals_1521, primals_1522, primals_1523, primals_1524, primals_1525, primals_1526, primals_1527, primals_1528, primals_1529, primals_1530, primals_1531, primals_1532, primals_1533, primals_1534, primals_1535, primals_1536, primals_1537, primals_1538, primals_1539, primals_1540, primals_1541, primals_1542, primals_1543, primals_1544, primals_1545, primals_1546, primals_1547, primals_1548, primals_1549, primals_1550, primals_1551, primals_1552, primals_1553, primals_1554, primals_1555, primals_1556, primals_1557, primals_1558, primals_1559, primals_1560, primals_1561, primals_1562, primals_1563, primals_1564, primals_1565, primals_1566, primals_1567, primals_1568, primals_1569, primals_1570, primals_1571, primals_1572, primals_1573, primals_1574, primals_1575, primals_1576, primals_1577, primals_1578, primals_1579, primals_1580, primals_1581, primals_1582, primals_1583, primals_1584, primals_1585, primals_1586, primals_1587, primals_1588, primals_1589, primals_1590, primals_1591, primals_1592, primals_1593, primals_1594, primals_1595, primals_1596, primals_1597, primals_1598, primals_1599, primals_1600, primals_1601, primals_1602, primals_1603, primals_1604, primals_1605, primals_1606, primals_1607, primals_1608, primals_1609, primals_1610, primals_1611, primals_1612, primals_1613, primals_1614, primals_1615, primals_1616, primals_1617, primals_1618, primals_1619, primals_1620, primals_1621, primals_1622, primals_1623, primals_1624, primals_1625, primals_1626, primals_1627, primals_1628, primals_1629, primals_1630, primals_1631, primals_1632, primals_1633, primals_1634, primals_1635, primals_1636, primals_1637, primals_1638, primals_1639, primals_1640, primals_1641, primals_1642, primals_1643, primals_1644, primals_1645, primals_1646, primals_1647, primals_1648, primals_1649, primals_1650, primals_1651, primals_1652, primals_1653, primals_1654, primals_1655, primals_1656, primals_1657, primals_1658, primals_1659, primals_1660, primals_1661, primals_1662, primals_1663, primals_1664, primals_1665, primals_1666, primals_1667, primals_1668, primals_1669, primals_1670, primals_1671, primals_1672, primals_1673, primals_1674, primals_1675, primals_1676, primals_1677, primals_1678, primals_1679, primals_1680, primals_1681, primals_1682, primals_1683, primals_1684, primals_1685, primals_1686, primals_1687, primals_1688, primals_1689, primals_1690, primals_1691, primals_1692, primals_1693, primals_1694, primals_1695, primals_1696, primals_1697, primals_1698, primals_1699, primals_1700, primals_1701, primals_1702, primals_1703, primals_1704, primals_1705, primals_1706, primals_1707, primals_1708, primals_1709, primals_1710, primals_1711, primals_1712, primals_1713, primals_1714, primals_1715, primals_1716, primals_1717, primals_1718, primals_1719, primals_1720, primals_1721, primals_1722, primals_1723, primals_1724, primals_1725, primals_1726, primals_1727, primals_1728, primals_1729, primals_1730, primals_1731, primals_1732, primals_1733])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
