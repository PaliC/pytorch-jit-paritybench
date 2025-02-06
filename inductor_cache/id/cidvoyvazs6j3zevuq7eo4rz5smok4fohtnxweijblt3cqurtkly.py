# AOT ID: ['16_forward']
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


# kernel path: inductor_cache/xd/cxdxqfemwglykm3yuxsfgmofblou6vcbihi7s6abznfqbq3knmvk.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_1 => mm_default_31
# Graph fragment:
#   %mm_default_31 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_1, %permute), kwargs = {})
triton_poi_fused_addmm_0 = async_compile.triton('triton_poi_fused_addmm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jv/cjvmtn5zwonbmaa2pqqykog57cx54oydn5qikqkdnnn4ceebyse2.py
# Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_4 => mm_default_29
# Graph fragment:
#   %mm_default_29 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_5, %permute), kwargs = {})
triton_poi_fused_addmm_1 = async_compile.triton('triton_poi_fused_addmm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fg/cfg3oxc2tci5qflsqrqz3tme7zhewhnd3cvsbngfgrtxlpqgb6m4.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_7 => mm_default_27
# Graph fragment:
#   %mm_default_27 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_9, %permute), kwargs = {})
triton_poi_fused_addmm_2 = async_compile.triton('triton_poi_fused_addmm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vb/cvb37lrt5ezyv6qjq3s3lz7j5mvxoepthd3an77ohzikb7f4cy4a.py
# Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_10 => mm_default_25
# Graph fragment:
#   %mm_default_25 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_13, %permute), kwargs = {})
triton_poi_fused_addmm_3 = async_compile.triton('triton_poi_fused_addmm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lu/cludkfj45gub22wv3gy2c5tx6f3uotpu2razhrjkklic66dscuoe.py
# Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_13 => mm_default_23
# Graph fragment:
#   %mm_default_23 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_17, %permute), kwargs = {})
triton_poi_fused_addmm_4 = async_compile.triton('triton_poi_fused_addmm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hf/chfcoevcsimmn2qenjfrzuje3z3niiiu7bsfy25je2jv6r2rhopm.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_16 => mm_default_21
# Graph fragment:
#   %mm_default_21 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_21, %permute), kwargs = {})
triton_poi_fused_addmm_5 = async_compile.triton('triton_poi_fused_addmm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mw/cmwytsyh7lfh7z5n4or4uw6ctm6inqmrmal4ojdz3iwyfqnilbvy.py
# Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_19 => mm_default_19
# Graph fragment:
#   %mm_default_19 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_25, %permute), kwargs = {})
triton_poi_fused_addmm_6 = async_compile.triton('triton_poi_fused_addmm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pg/cpgtdfn4dyklgobmv3dy5fuja7qxehn7icvlovazfxppbm2pfk7s.py
# Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_22 => mm_default_17
# Graph fragment:
#   %mm_default_17 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_29, %permute), kwargs = {})
triton_poi_fused_addmm_7 = async_compile.triton('triton_poi_fused_addmm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ub/cubxdvgq6wx77aift6uypp4xygiznbnhufzdsk7f42yzxhlj443u.py
# Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_31 => mm_default_11
# Graph fragment:
#   %mm_default_11 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_41, %permute), kwargs = {})
triton_poi_fused_addmm_8 = async_compile.triton('triton_poi_fused_addmm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yf/cyfujbmzswbxiu3uhxoozlcy7task3ejil6gbvtxmtc25tefa6fb.py
# Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_34 => mm_default_9
# Graph fragment:
#   %mm_default_9 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_45, %permute), kwargs = {})
triton_poi_fused_addmm_9 = async_compile.triton('triton_poi_fused_addmm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a6/ca6y53wiijimpfpnuglpbtw7xgo6s7xv6c5paj5ivn6e6tzwpszs.py
# Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_37 => mm_default_7
# Graph fragment:
#   %mm_default_7 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_49, %permute), kwargs = {})
triton_poi_fused_addmm_10 = async_compile.triton('triton_poi_fused_addmm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s7/cs7xdgwvemczkn5efxbbzlhjri6sw5xcuedvfuihmsa337fsk3di.py
# Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_40 => mm_default_5
# Graph fragment:
#   %mm_default_5 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_53, %permute), kwargs = {})
triton_poi_fused_addmm_11 = async_compile.triton('triton_poi_fused_addmm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m4/cm4v4fiikbt56oeuklvu77mmkpt74kvwbjqfdjegx7hyggisupxk.py
# Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_25 => mm_default_15
# Graph fragment:
#   %mm_default_15 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_33, %permute), kwargs = {})
triton_poi_fused_addmm_12 = async_compile.triton('triton_poi_fused_addmm_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tm/ctm5orxe4rhxm6z7ctuhgficbef7anqhplsv4xlubpzkw2mjauma.py
# Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_28 => mm_default_13
# Graph fragment:
#   %mm_default_13 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_37, %permute), kwargs = {})
triton_poi_fused_addmm_13 = async_compile.triton('triton_poi_fused_addmm_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cc/ccckemhfgvloc4s3glmq6jn5ph4lkquvzgunbphbkjkile3oe2cr.py
# Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_43 => mm_default_3
# Graph fragment:
#   %mm_default_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_57, %permute), kwargs = {})
triton_poi_fused_addmm_14 = async_compile.triton('triton_poi_fused_addmm_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m4/cm4veopghrhkxcjj2mcafeedgjgj5egkagf2w24ajhfh6veczdu5.py
# Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   input_46 => mm_default_1
# Graph fragment:
#   %mm_default_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%select_61, %permute), kwargs = {})
triton_poi_fused_addmm_15 = async_compile.triton('triton_poi_fused_addmm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pb/cpbfbsvav62lw2mxgmwxyzt5jssacngtkaedtlnlco3aagyfljcs.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_4, input_5, input_7, input_8, input_10, input_11, input_13, input_14, input_16, input_17, input_19, input_20], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_tensor_31
#   input_10 => add_tensor_25
#   input_11 => relu_3
#   input_13 => add_tensor_23
#   input_14 => relu_4
#   input_16 => add_tensor_21
#   input_17 => relu_5
#   input_19 => add_tensor_19
#   input_2 => relu
#   input_20 => relu_6
#   input_4 => add_tensor_29
#   input_5 => relu_1
#   input_7 => add_tensor_27
#   input_8 => relu_2
# Graph fragment:
#   %add_tensor_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_31, %primals_3), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_31,), kwargs = {})
#   %add_tensor_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_29, %primals_3), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_29,), kwargs = {})
#   %add_tensor_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_27, %primals_3), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_27,), kwargs = {})
#   %add_tensor_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_25, %primals_3), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_25,), kwargs = {})
#   %add_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_23, %primals_3), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_23,), kwargs = {})
#   %add_tensor_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_21, %primals_3), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_21,), kwargs = {})
#   %add_tensor_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_19, %primals_3), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_19,), kwargs = {})
triton_poi_fused_addmm_relu_16 = async_compile.triton('triton_poi_fused_addmm_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_16', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_16(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp8 = tl.load(in_out_ptr2 + (x2), xmask)
    tmp11 = tl.load(in_out_ptr3 + (x2), xmask)
    tmp14 = tl.load(in_out_ptr4 + (x2), xmask)
    tmp17 = tl.load(in_out_ptr5 + (x2), xmask)
    tmp20 = tl.load(in_out_ptr6 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tmp9 = tmp8 + tmp1
    tmp10 = triton_helpers.maximum(tmp3, tmp9)
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp3, tmp12)
    tmp15 = tmp14 + tmp1
    tmp16 = triton_helpers.maximum(tmp3, tmp15)
    tmp18 = tmp17 + tmp1
    tmp19 = triton_helpers.maximum(tmp3, tmp18)
    tmp21 = tmp20 + tmp1
    tmp22 = triton_helpers.maximum(tmp3, tmp21)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr1 + (x2), tmp7, xmask)
    tl.store(in_out_ptr2 + (x2), tmp10, xmask)
    tl.store(in_out_ptr3 + (x2), tmp13, xmask)
    tl.store(in_out_ptr4 + (x2), tmp16, xmask)
    tl.store(in_out_ptr5 + (x2), tmp19, xmask)
    tl.store(in_out_ptr6 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/iz/cizmvcpu2nprujkufmizdnt2sjcvzypz3a7o2cvv3ksu4bwttquu.py
# Topologically Sorted Source Nodes: [input_43, input_44, input_46, input_47], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_43 => add_tensor_3
#   input_44 => relu_14
#   input_46 => add_tensor_1
#   input_47 => relu_15
# Graph fragment:
#   %add_tensor_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_3, %primals_3), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_3,), kwargs = {})
#   %add_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_1, %primals_3), kwargs = {})
#   %relu_15 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_1,), kwargs = {})
triton_poi_fused_addmm_relu_17 = async_compile.triton('triton_poi_fused_addmm_relu_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_17(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = tmp5 + tmp1
    tmp7 = triton_helpers.maximum(tmp3, tmp6)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr1 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cp/ccpl2hmmusxt4km5dnc2nfmtssasksodzostdefvoifsm4m54thi.py
# Topologically Sorted Source Nodes: [input_3, anchor_xy_1, input_6, anchor_xy_3, input_9, anchor_xy_5, input_12, anchor_xy_7, input_15, anchor_xy_9, input_18, anchor_xy_11, input_21, anchor_xy_13], Original ATen: [aten.addmm, aten.add]
# Source node to ATen node mapping:
#   anchor_xy_1 => add
#   anchor_xy_11 => add_5
#   anchor_xy_13 => add_6
#   anchor_xy_3 => add_1
#   anchor_xy_5 => add_2
#   anchor_xy_7 => add_3
#   anchor_xy_9 => add_4
#   input_12 => add_tensor_24
#   input_15 => add_tensor_22
#   input_18 => add_tensor_20
#   input_21 => add_tensor_18
#   input_3 => add_tensor_30
#   input_6 => add_tensor_28
#   input_9 => add_tensor_26
# Graph fragment:
#   %add_tensor_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_30, %primals_5), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_1, %add_tensor_30), kwargs = {})
#   %add_tensor_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_28, %primals_5), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_5, %add_tensor_28), kwargs = {})
#   %add_tensor_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_26, %primals_5), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_9, %add_tensor_26), kwargs = {})
#   %add_tensor_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_24, %primals_5), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_13, %add_tensor_24), kwargs = {})
#   %add_tensor_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_22, %primals_5), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_17, %add_tensor_22), kwargs = {})
#   %add_tensor_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_20, %primals_5), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_21, %add_tensor_20), kwargs = {})
#   %add_tensor_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_18, %primals_5), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_25, %add_tensor_18), kwargs = {})
triton_poi_fused_add_addmm_18 = async_compile.triton('triton_poi_fused_add_addmm_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_18', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_18(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp9 = tl.load(in_ptr0 + (2 + 16*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr2 + (x2), xmask)
    tmp13 = tl.load(in_ptr0 + (3 + 16*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_out_ptr3 + (x2), xmask)
    tmp17 = tl.load(in_ptr0 + (4 + 16*x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_out_ptr4 + (x2), xmask)
    tmp21 = tl.load(in_ptr0 + (5 + 16*x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr5 + (x2), xmask)
    tmp25 = tl.load(in_ptr0 + (6 + 16*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr6 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp6 + tmp2
    tmp8 = tmp5 + tmp7
    tmp11 = tmp10 + tmp2
    tmp12 = tmp9 + tmp11
    tmp15 = tmp14 + tmp2
    tmp16 = tmp13 + tmp15
    tmp19 = tmp18 + tmp2
    tmp20 = tmp17 + tmp19
    tmp23 = tmp22 + tmp2
    tmp24 = tmp21 + tmp23
    tmp27 = tmp26 + tmp2
    tmp28 = tmp25 + tmp27
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr1 + (x2), tmp8, xmask)
    tl.store(in_out_ptr2 + (x2), tmp12, xmask)
    tl.store(in_out_ptr3 + (x2), tmp16, xmask)
    tl.store(in_out_ptr4 + (x2), tmp20, xmask)
    tl.store(in_out_ptr5 + (x2), tmp24, xmask)
    tl.store(in_out_ptr6 + (x2), tmp28, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g2/cg2dvj57j5txzxxmatnqw5nuarklr5eyxvrjbutmecg2iwknudjm.py
# Topologically Sorted Source Nodes: [input_24, anchor_xy_15, input_27, anchor_xy_17, input_30, anchor_xy_19, input_33, anchor_xy_21, input_36, anchor_xy_23, input_39, anchor_xy_25, input_42, anchor_xy_27], Original ATen: [aten.addmm, aten.add]
# Source node to ATen node mapping:
#   anchor_xy_15 => add_7
#   anchor_xy_17 => add_8
#   anchor_xy_19 => add_9
#   anchor_xy_21 => add_10
#   anchor_xy_23 => add_11
#   anchor_xy_25 => add_12
#   anchor_xy_27 => add_13
#   input_24 => add_tensor_16
#   input_27 => add_tensor_14
#   input_30 => add_tensor_12
#   input_33 => add_tensor_10
#   input_36 => add_tensor_8
#   input_39 => add_tensor_6
#   input_42 => add_tensor_4
# Graph fragment:
#   %add_tensor_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_16, %primals_5), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_29, %add_tensor_16), kwargs = {})
#   %add_tensor_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_14, %primals_5), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_33, %add_tensor_14), kwargs = {})
#   %add_tensor_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_12, %primals_5), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_37, %add_tensor_12), kwargs = {})
#   %add_tensor_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_10, %primals_5), kwargs = {})
#   %add_10 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_41, %add_tensor_10), kwargs = {})
#   %add_tensor_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_8, %primals_5), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_45, %add_tensor_8), kwargs = {})
#   %add_tensor_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_6, %primals_5), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_49, %add_tensor_6), kwargs = {})
#   %add_tensor_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_4, %primals_5), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_53, %add_tensor_4), kwargs = {})
triton_poi_fused_add_addmm_19 = async_compile.triton('triton_poi_fused_add_addmm_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_out_ptr4': '*fp32', 'in_out_ptr5': '*fp32', 'in_out_ptr6': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 15, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_19(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (7 + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (8 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp9 = tl.load(in_ptr0 + (9 + 16*x2), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_out_ptr2 + (x2), xmask)
    tmp13 = tl.load(in_ptr0 + (10 + 16*x2), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_out_ptr3 + (x2), xmask)
    tmp17 = tl.load(in_ptr0 + (11 + 16*x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_out_ptr4 + (x2), xmask)
    tmp21 = tl.load(in_ptr0 + (12 + 16*x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_out_ptr5 + (x2), xmask)
    tmp25 = tl.load(in_ptr0 + (13 + 16*x2), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_out_ptr6 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp6 + tmp2
    tmp8 = tmp5 + tmp7
    tmp11 = tmp10 + tmp2
    tmp12 = tmp9 + tmp11
    tmp15 = tmp14 + tmp2
    tmp16 = tmp13 + tmp15
    tmp19 = tmp18 + tmp2
    tmp20 = tmp17 + tmp19
    tmp23 = tmp22 + tmp2
    tmp24 = tmp21 + tmp23
    tmp27 = tmp26 + tmp2
    tmp28 = tmp25 + tmp27
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr1 + (x2), tmp8, xmask)
    tl.store(in_out_ptr2 + (x2), tmp12, xmask)
    tl.store(in_out_ptr3 + (x2), tmp16, xmask)
    tl.store(in_out_ptr4 + (x2), tmp20, xmask)
    tl.store(in_out_ptr5 + (x2), tmp24, xmask)
    tl.store(in_out_ptr6 + (x2), tmp28, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qr/cqrrmmfoc7ndcdvntpelv2fjfecoqmx5wttllesbsby234dq7hab.py
# Topologically Sorted Source Nodes: [input_45, anchor_xy_29, input_48, anchor_xy_31], Original ATen: [aten.addmm, aten.add]
# Source node to ATen node mapping:
#   anchor_xy_29 => add_14
#   anchor_xy_31 => add_15
#   input_45 => add_tensor_2
#   input_48 => add_tensor
# Graph fragment:
#   %add_tensor_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_2, %primals_5), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_57, %add_tensor_2), kwargs = {})
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_5), kwargs = {})
#   %add_15 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%select_61, %add_tensor), kwargs = {})
triton_poi_fused_add_addmm_20 = async_compile.triton('triton_poi_fused_add_addmm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_addmm_20', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_addmm_20(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (14 + 16*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp6 + tmp2
    tmp8 = tmp5 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(in_out_ptr1 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3w/c3wyhvt6afypl3vmmyem2xg4yp3cxfwrl464ndwtx4gz6dpfb42b.py
# Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_1 => getitem_1
# Graph fragment:
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%max_1, 1), kwargs = {})
triton_poi_fused_max_21 = async_compile.triton('triton_poi_fused_max_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_21(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 > tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = tmp0 != tmp0
    tmp5 = tmp1 != tmp1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp2 | tmp6
    tmp8 = tmp4 & tmp5
    tmp9 = tmp3 | tmp8
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tmp7 | tmp13
    tmp15 = tl.where(tmp14, tmp0, tmp1)
    tmp16 = tl.where(tmp14, tmp10, tmp11)
    tmp18 = tmp15 > tmp17
    tmp19 = tmp15 == tmp17
    tmp20 = tmp15 != tmp15
    tmp21 = tmp17 != tmp17
    tmp22 = tmp20 > tmp21
    tmp23 = tmp18 | tmp22
    tmp24 = tmp20 & tmp21
    tmp25 = tmp19 | tmp24
    tmp26 = tl.full([1], 2, tl.int64)
    tmp27 = tmp16 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tmp23 | tmp28
    tmp30 = tl.where(tmp29, tmp15, tmp17)
    tmp31 = tl.where(tmp29, tmp16, tmp26)
    tmp33 = tmp30 > tmp32
    tmp34 = tmp30 == tmp32
    tmp35 = tmp30 != tmp30
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tl.full([1], 3, tl.int64)
    tmp42 = tmp31 < tmp41
    tmp43 = tmp40 & tmp42
    tmp44 = tmp38 | tmp43
    tmp45 = tl.where(tmp44, tmp30, tmp32)
    tmp46 = tl.where(tmp44, tmp31, tmp41)
    tl.store(out_ptr0 + (x0), tmp46, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxttnsailrvx7uema5f4luoialnxpishr5rga7r7he3hu5lop5y.py
# Topologically Sorted Source Nodes: [max_1, logits_1], Original ATen: [aten.max, aten.sub]
# Source node to ATen node mapping:
#   logits_1 => sub
#   max_1 => max_1
# Graph fragment:
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%mm_1, 1, True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mm_1, %getitem), kwargs = {})
triton_poi_fused_max_sub_22 = async_compile.triton('triton_poi_fused_max_sub_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_sub_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_sub_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_0.run(primals_1, buf0, 16, grid=grid(16), stream=stream0)
        buf10 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_1.run(primals_1, buf10, 16, grid=grid(16), stream=stream0)
        buf20 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_2.run(primals_1, buf20, 16, grid=grid(16), stream=stream0)
        buf30 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_3.run(primals_1, buf30, 16, grid=grid(16), stream=stream0)
        buf40 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_4.run(primals_1, buf40, 16, grid=grid(16), stream=stream0)
        buf50 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_5.run(primals_1, buf50, 16, grid=grid(16), stream=stream0)
        buf60 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_6.run(primals_1, buf60, 16, grid=grid(16), stream=stream0)
        buf6 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_0.run(primals_7, buf6, 16, grid=grid(16), stream=stream0)
        buf16 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_2], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_1.run(primals_7, buf16, 16, grid=grid(16), stream=stream0)
        buf26 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_4], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_2.run(primals_7, buf26, 16, grid=grid(16), stream=stream0)
        buf36 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_6], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_3.run(primals_7, buf36, 16, grid=grid(16), stream=stream0)
        buf46 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_8], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_4.run(primals_7, buf46, 16, grid=grid(16), stream=stream0)
        buf56 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_10], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_5.run(primals_7, buf56, 16, grid=grid(16), stream=stream0)
        buf66 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_12], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_6.run(primals_7, buf66, 16, grid=grid(16), stream=stream0)
        buf70 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_7.run(primals_1, buf70, 16, grid=grid(16), stream=stream0)
        buf100 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_8.run(primals_1, buf100, 16, grid=grid(16), stream=stream0)
        buf110 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_9.run(primals_1, buf110, 16, grid=grid(16), stream=stream0)
        buf120 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_10.run(primals_1, buf120, 16, grid=grid(16), stream=stream0)
        buf130 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_11.run(primals_1, buf130, 16, grid=grid(16), stream=stream0)
        buf80 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_12.run(primals_1, buf80, 16, grid=grid(16), stream=stream0)
        buf90 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_13.run(primals_1, buf90, 16, grid=grid(16), stream=stream0)
        buf76 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_14], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_7.run(primals_7, buf76, 16, grid=grid(16), stream=stream0)
        buf86 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_16], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_12.run(primals_7, buf86, 16, grid=grid(16), stream=stream0)
        buf96 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_18], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_13.run(primals_7, buf96, 16, grid=grid(16), stream=stream0)
        buf106 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_20], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_8.run(primals_7, buf106, 16, grid=grid(16), stream=stream0)
        buf116 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_22], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_9.run(primals_7, buf116, 16, grid=grid(16), stream=stream0)
        buf126 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_24], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_10.run(primals_7, buf126, 16, grid=grid(16), stream=stream0)
        buf136 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_26], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_11.run(primals_7, buf136, 16, grid=grid(16), stream=stream0)
        buf140 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_14.run(primals_1, buf140, 16, grid=grid(16), stream=stream0)
        buf150 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.addmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_15.run(primals_1, buf150, 16, grid=grid(16), stream=stream0)
        buf146 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_28], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_14.run(primals_7, buf146, 16, grid=grid(16), stream=stream0)
        buf156 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits_30], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_15.run(primals_7, buf156, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf1)
        buf11 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.mm(buf10, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf11)
        buf21 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf20, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf21)
        buf31 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.addmm]
        extern_kernels.mm(buf30, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf31)
        buf41 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.addmm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf41)
        buf51 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.addmm]
        extern_kernels.mm(buf50, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf51)
        buf61 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.addmm]
        extern_kernels.mm(buf60, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf61)
        buf71 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [input_22], Original ATen: [aten.addmm]
        extern_kernels.mm(buf70, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf71)
        buf101 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.addmm]
        extern_kernels.mm(buf100, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf101)
        buf111 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.addmm]
        extern_kernels.mm(buf110, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf111)
        buf121 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.addmm]
        extern_kernels.mm(buf120, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf121)
        buf131 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [input_40], Original ATen: [aten.addmm]
        extern_kernels.mm(buf130, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf131)
        buf81 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [input_25], Original ATen: [aten.addmm]
        extern_kernels.mm(buf80, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf81)
        buf91 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.addmm]
        extern_kernels.mm(buf90, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf91)
        buf141 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.addmm]
        extern_kernels.mm(buf140, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf141)
        buf151 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.addmm]
        extern_kernels.mm(buf150, reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf151)
        del primals_2
        buf2 = buf1; del buf1  # reuse
        buf12 = buf11; del buf11  # reuse
        buf22 = buf21; del buf21  # reuse
        buf32 = buf31; del buf31  # reuse
        buf42 = buf41; del buf41  # reuse
        buf52 = buf51; del buf51  # reuse
        buf62 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2, input_4, input_5, input_7, input_8, input_10, input_11, input_13, input_14, input_16, input_17, input_19, input_20], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_16.run(buf2, buf12, buf22, buf32, buf42, buf52, buf62, primals_3, 16, grid=grid(16), stream=stream0)
        buf72 = buf71; del buf71  # reuse
        buf82 = buf81; del buf81  # reuse
        buf92 = buf91; del buf91  # reuse
        buf102 = buf101; del buf101  # reuse
        buf112 = buf111; del buf111  # reuse
        buf122 = buf121; del buf121  # reuse
        buf132 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [input_22, input_23, input_25, input_26, input_28, input_29, input_31, input_32, input_34, input_35, input_37, input_38, input_40, input_41], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_16.run(buf72, buf82, buf92, buf102, buf112, buf122, buf132, primals_3, 16, grid=grid(16), stream=stream0)
        buf142 = buf141; del buf141  # reuse
        buf152 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44, input_46, input_47], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_17.run(buf142, buf152, primals_3, 16, grid=grid(16), stream=stream0)
        del primals_3
        buf3 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.mm(buf2, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf3)
        buf13 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf13)
        buf23 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf22, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf23)
        buf33 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.addmm]
        extern_kernels.mm(buf32, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf33)
        buf43 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.addmm]
        extern_kernels.mm(buf42, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf43)
        buf53 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.addmm]
        extern_kernels.mm(buf52, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf53)
        buf63 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.addmm]
        extern_kernels.mm(buf62, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf63)
        buf73 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.addmm]
        extern_kernels.mm(buf72, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf73)
        buf103 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.addmm]
        extern_kernels.mm(buf102, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf103)
        buf113 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.addmm]
        extern_kernels.mm(buf112, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf113)
        buf123 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.addmm]
        extern_kernels.mm(buf122, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf123)
        buf133 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_42], Original ATen: [aten.addmm]
        extern_kernels.mm(buf132, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf133)
        buf83 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_27], Original ATen: [aten.addmm]
        extern_kernels.mm(buf82, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf83)
        buf93 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_30], Original ATen: [aten.addmm]
        extern_kernels.mm(buf92, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf93)
        buf143 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_45], Original ATen: [aten.addmm]
        extern_kernels.mm(buf142, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf143)
        buf153 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.addmm]
        extern_kernels.mm(buf152, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf153)
        buf4 = buf3; del buf3  # reuse
        buf14 = buf13; del buf13  # reuse
        buf24 = buf23; del buf23  # reuse
        buf34 = buf33; del buf33  # reuse
        buf44 = buf43; del buf43  # reuse
        buf54 = buf53; del buf53  # reuse
        buf64 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_3, anchor_xy_1, input_6, anchor_xy_3, input_9, anchor_xy_5, input_12, anchor_xy_7, input_15, anchor_xy_9, input_18, anchor_xy_11, input_21, anchor_xy_13], Original ATen: [aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_18.run(buf4, buf14, buf24, buf34, buf44, buf54, buf64, primals_1, primals_5, 16, grid=grid(16), stream=stream0)
        buf74 = buf73; del buf73  # reuse
        buf84 = buf83; del buf83  # reuse
        buf94 = buf93; del buf93  # reuse
        buf104 = buf103; del buf103  # reuse
        buf114 = buf113; del buf113  # reuse
        buf124 = buf123; del buf123  # reuse
        buf134 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [input_24, anchor_xy_15, input_27, anchor_xy_17, input_30, anchor_xy_19, input_33, anchor_xy_21, input_36, anchor_xy_23, input_39, anchor_xy_25, input_42, anchor_xy_27], Original ATen: [aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_19.run(buf74, buf84, buf94, buf104, buf114, buf124, buf134, primals_1, primals_5, 16, grid=grid(16), stream=stream0)
        buf144 = buf143; del buf143  # reuse
        buf154 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [input_45, anchor_xy_29, input_48, anchor_xy_31], Original ATen: [aten.addmm, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_20.run(buf144, buf154, primals_1, primals_5, 16, grid=grid(16), stream=stream0)
        del primals_5
        buf5 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pred_xy], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf5)
        buf7 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(buf5, buf6, out=buf7)
        buf15 = reinterpret_tensor(buf6, (4, 4), (4, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf14, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf15)
        buf17 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [logits_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf15, buf16, out=buf17)
        buf25 = reinterpret_tensor(buf16, (4, 4), (4, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf24, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf25)
        buf27 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [logits_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf25, buf26, out=buf27)
        buf35 = reinterpret_tensor(buf26, (4, 4), (4, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf34, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf35)
        buf37 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [logits_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf35, buf36, out=buf37)
        buf45 = reinterpret_tensor(buf36, (4, 4), (4, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf44, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf45)
        buf47 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [logits_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf45, buf46, out=buf47)
        buf55 = reinterpret_tensor(buf46, (4, 4), (4, 1), 0); del buf46  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf54, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf55)
        buf57 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [logits_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf55, buf56, out=buf57)
        buf65 = reinterpret_tensor(buf56, (4, 4), (4, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf64, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf65)
        buf67 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [logits_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf65, buf66, out=buf67)
        buf75 = reinterpret_tensor(buf66, (4, 4), (4, 1), 0); del buf66  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf74, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf75)
        buf77 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [logits_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf75, buf76, out=buf77)
        buf85 = reinterpret_tensor(buf76, (4, 4), (4, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf84, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf85)
        buf87 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [logits_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf85, buf86, out=buf87)
        buf95 = reinterpret_tensor(buf86, (4, 4), (4, 1), 0); del buf86  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf94, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf95)
        buf97 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [logits_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf95, buf96, out=buf97)
        buf105 = reinterpret_tensor(buf96, (4, 4), (4, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf104, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf105)
        buf107 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [logits_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf105, buf106, out=buf107)
        buf115 = reinterpret_tensor(buf106, (4, 4), (4, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf114, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf115)
        buf117 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [logits_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf115, buf116, out=buf117)
        buf125 = reinterpret_tensor(buf116, (4, 4), (4, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf124, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf125)
        buf127 = buf115; del buf115  # reuse
        # Topologically Sorted Source Nodes: [logits_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf125, buf126, out=buf127)
        buf135 = reinterpret_tensor(buf126, (4, 4), (4, 1), 0); del buf126  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf134, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf135)
        buf137 = buf125; del buf125  # reuse
        # Topologically Sorted Source Nodes: [logits_26], Original ATen: [aten.mm]
        extern_kernels.mm(buf135, buf136, out=buf137)
        buf145 = reinterpret_tensor(buf136, (4, 4), (4, 1), 0); del buf136  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf144, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf145)
        buf147 = buf135; del buf135  # reuse
        # Topologically Sorted Source Nodes: [logits_28], Original ATen: [aten.mm]
        extern_kernels.mm(buf145, buf146, out=buf147)
        buf155 = reinterpret_tensor(buf146, (4, 4), (4, 1), 0); del buf146  # reuse
        # Topologically Sorted Source Nodes: [pred_xy_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf154, reinterpret_tensor(primals_6, (4, 4), (1, 4), 0), out=buf155)
        buf157 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [logits_30], Original ATen: [aten.mm]
        extern_kernels.mm(buf155, buf156, out=buf157)
        del buf155
        buf8 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf7, buf8, 4, grid=grid(4), stream=stream0)
        buf9 = reinterpret_tensor(buf156, (4, 4), (4, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [max_1, logits_1], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf7, buf9, 16, grid=grid(16), stream=stream0)
        buf18 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_2], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf17, buf18, 4, grid=grid(4), stream=stream0)
        buf19 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [max_2, logits_3], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf17, buf19, 16, grid=grid(16), stream=stream0)
        buf28 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_3], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf27, buf28, 4, grid=grid(4), stream=stream0)
        buf29 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [max_3, logits_5], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf27, buf29, 16, grid=grid(16), stream=stream0)
        buf38 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_4], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf37, buf38, 4, grid=grid(4), stream=stream0)
        buf39 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [max_4, logits_7], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf37, buf39, 16, grid=grid(16), stream=stream0)
        buf48 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_5], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf47, buf48, 4, grid=grid(4), stream=stream0)
        buf49 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [max_5, logits_9], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf47, buf49, 16, grid=grid(16), stream=stream0)
        buf58 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_6], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf57, buf58, 4, grid=grid(4), stream=stream0)
        buf59 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [max_6, logits_11], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf57, buf59, 16, grid=grid(16), stream=stream0)
        buf68 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_7], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf67, buf68, 4, grid=grid(4), stream=stream0)
        buf69 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [max_7, logits_13], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf67, buf69, 16, grid=grid(16), stream=stream0)
        buf78 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_8], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf77, buf78, 4, grid=grid(4), stream=stream0)
        buf79 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [max_8, logits_15], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf77, buf79, 16, grid=grid(16), stream=stream0)
        buf88 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_9], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf87, buf88, 4, grid=grid(4), stream=stream0)
        buf89 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [max_9, logits_17], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf87, buf89, 16, grid=grid(16), stream=stream0)
        buf98 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_10], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf97, buf98, 4, grid=grid(4), stream=stream0)
        buf99 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [max_10, logits_19], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf97, buf99, 16, grid=grid(16), stream=stream0)
        buf108 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_11], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf107, buf108, 4, grid=grid(4), stream=stream0)
        buf109 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [max_11, logits_21], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf107, buf109, 16, grid=grid(16), stream=stream0)
        buf118 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_12], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf117, buf118, 4, grid=grid(4), stream=stream0)
        buf119 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [max_12, logits_23], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf117, buf119, 16, grid=grid(16), stream=stream0)
        buf128 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_13], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf127, buf128, 4, grid=grid(4), stream=stream0)
        buf129 = buf117; del buf117  # reuse
        # Topologically Sorted Source Nodes: [max_13, logits_25], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf127, buf129, 16, grid=grid(16), stream=stream0)
        buf138 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_14], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf137, buf138, 4, grid=grid(4), stream=stream0)
        buf139 = buf127; del buf127  # reuse
        # Topologically Sorted Source Nodes: [max_14, logits_27], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf137, buf139, 16, grid=grid(16), stream=stream0)
        buf148 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_15], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf147, buf148, 4, grid=grid(4), stream=stream0)
        buf149 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [max_15, logits_29], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf147, buf149, 16, grid=grid(16), stream=stream0)
        buf158 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_16], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_21.run(buf157, buf158, 4, grid=grid(4), stream=stream0)
        buf159 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [max_16, logits_31], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_22.run(buf157, buf159, 16, grid=grid(16), stream=stream0)
        del buf157
    return (buf9, buf19, buf29, buf39, buf49, buf59, buf69, buf79, buf89, buf99, buf109, buf119, buf129, buf139, buf149, buf159, reinterpret_tensor(primals_1, (4, 4), (64, 16), 0), buf2, buf4, buf8, reinterpret_tensor(primals_1, (4, 4), (64, 16), 1), buf12, buf14, buf18, reinterpret_tensor(primals_1, (4, 4), (64, 16), 2), buf22, buf24, buf28, reinterpret_tensor(primals_1, (4, 4), (64, 16), 3), buf32, buf34, buf38, reinterpret_tensor(primals_1, (4, 4), (64, 16), 4), buf42, buf44, buf48, reinterpret_tensor(primals_1, (4, 4), (64, 16), 5), buf52, buf54, buf58, reinterpret_tensor(primals_1, (4, 4), (64, 16), 6), buf62, buf64, buf68, reinterpret_tensor(primals_1, (4, 4), (64, 16), 7), buf72, buf74, buf78, reinterpret_tensor(primals_1, (4, 4), (64, 16), 8), buf82, buf84, buf88, reinterpret_tensor(primals_1, (4, 4), (64, 16), 9), buf92, buf94, buf98, reinterpret_tensor(primals_1, (4, 4), (64, 16), 10), buf102, buf104, buf108, reinterpret_tensor(primals_1, (4, 4), (64, 16), 11), buf112, buf114, buf118, reinterpret_tensor(primals_1, (4, 4), (64, 16), 12), buf122, buf124, buf128, reinterpret_tensor(primals_1, (4, 4), (64, 16), 13), buf132, buf134, buf138, reinterpret_tensor(primals_1, (4, 4), (64, 16), 14), buf142, buf144, buf148, reinterpret_tensor(primals_1, (4, 4), (64, 16), 15), buf152, buf154, buf158, reinterpret_tensor(primals_7, (4, 4), (64, 16), 15), primals_6, primals_4, reinterpret_tensor(primals_7, (4, 4), (64, 16), 14), reinterpret_tensor(primals_7, (4, 4), (64, 16), 13), reinterpret_tensor(primals_7, (4, 4), (64, 16), 12), reinterpret_tensor(primals_7, (4, 4), (64, 16), 11), reinterpret_tensor(primals_7, (4, 4), (64, 16), 10), reinterpret_tensor(primals_7, (4, 4), (64, 16), 9), reinterpret_tensor(primals_7, (4, 4), (64, 16), 8), reinterpret_tensor(primals_7, (4, 4), (64, 16), 7), reinterpret_tensor(primals_7, (4, 4), (64, 16), 6), reinterpret_tensor(primals_7, (4, 4), (64, 16), 5), reinterpret_tensor(primals_7, (4, 4), (64, 16), 4), reinterpret_tensor(primals_7, (4, 4), (64, 16), 3), reinterpret_tensor(primals_7, (4, 4), (64, 16), 2), reinterpret_tensor(primals_7, (4, 4), (64, 16), 1), reinterpret_tensor(primals_7, (4, 4), (64, 16), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
