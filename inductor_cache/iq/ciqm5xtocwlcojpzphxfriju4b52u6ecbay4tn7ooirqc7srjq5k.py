# AOT ID: ['7_inference']
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


# kernel path: inductor_cache/r6/cr6hvd5vwvnlwwt75vitryq4wz27itihmmbm5sdv7ukdczq4soas.py
# Topologically Sorted Source Nodes: [max_1, max_2], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_1 => max_1
#   max_2 => max_2
# Graph fragment:
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%select, 2), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%getitem, 1), kwargs = {})
triton_poi_fused_max_0 = async_compile.triton('triton_poi_fused_max_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (64*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 64*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 64*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 64*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4 + 64*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (5 + 64*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (6 + 64*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (7 + 64*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (8 + 64*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (9 + 64*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (10 + 64*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (11 + 64*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (12 + 64*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (13 + 64*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (14 + 64*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (15 + 64*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = triton_helpers.maximum(tmp6, tmp13)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp14, tmp21)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp29 = triton_helpers.maximum(tmp27, tmp28)
    tmp30 = triton_helpers.maximum(tmp22, tmp29)
    tl.store(out_ptr0 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fe/cfenxv7ys2emkjefodjeq75dyrouzax43dn5g2vqdtl6u2mjtdw6.py
# Topologically Sorted Source Nodes: [max_3, max_4], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_3 => max_3
#   max_4 => max_4
# Graph fragment:
#   %max_3 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%select_9, 2), kwargs = {})
#   %max_4 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%getitem_4, 1), kwargs = {})
triton_poi_fused_max_1 = async_compile.triton('triton_poi_fused_max_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + 64*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (17 + 64*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (18 + 64*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (19 + 64*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (20 + 64*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (21 + 64*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (22 + 64*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (23 + 64*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (24 + 64*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (25 + 64*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (26 + 64*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (27 + 64*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (28 + 64*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (29 + 64*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (30 + 64*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (31 + 64*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = triton_helpers.maximum(tmp6, tmp13)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp14, tmp21)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp29 = triton_helpers.maximum(tmp27, tmp28)
    tmp30 = triton_helpers.maximum(tmp22, tmp29)
    tl.store(out_ptr0 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ow/cow5x4lxo3cish3w5uazjwampcyf6jvrx6mx3nury42xc3jkbtzu.py
# Topologically Sorted Source Nodes: [max_5, max_6], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_5 => max_5
#   max_6 => max_6
# Graph fragment:
#   %max_5 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%select_21, 2), kwargs = {})
#   %max_6 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%getitem_8, 1), kwargs = {})
triton_poi_fused_max_2 = async_compile.triton('triton_poi_fused_max_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + 64*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (33 + 64*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (34 + 64*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (35 + 64*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (36 + 64*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (37 + 64*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (38 + 64*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (39 + 64*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (40 + 64*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (41 + 64*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (42 + 64*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (43 + 64*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (44 + 64*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (45 + 64*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (46 + 64*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (47 + 64*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = triton_helpers.maximum(tmp6, tmp13)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp14, tmp21)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp29 = triton_helpers.maximum(tmp27, tmp28)
    tmp30 = triton_helpers.maximum(tmp22, tmp29)
    tl.store(out_ptr0 + (x0), tmp30, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pl/cpl3gp6ok7vtzirq5wgna7p6bmlrhadtyjmdsprlwfs3dp77gk2p.py
# Topologically Sorted Source Nodes: [max_7, max_8], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_7 => max_7
#   max_8 => max_8
# Graph fragment:
#   %max_7 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%select_33, 2), kwargs = {})
#   %max_8 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%getitem_12, 1), kwargs = {})
triton_poi_fused_max_3 = async_compile.triton('triton_poi_fused_max_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (48 + 64*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (49 + 64*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (50 + 64*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (51 + 64*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (52 + 64*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (53 + 64*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (54 + 64*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (55 + 64*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (56 + 64*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (57 + 64*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (58 + 64*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr0 + (59 + 64*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (60 + 64*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (61 + 64*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr0 + (62 + 64*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (63 + 64*x0), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp9 = triton_helpers.maximum(tmp7, tmp8)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp13 = triton_helpers.maximum(tmp11, tmp12)
    tmp14 = triton_helpers.maximum(tmp6, tmp13)
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp21 = triton_helpers.maximum(tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp14, tmp21)
    tmp25 = triton_helpers.maximum(tmp23, tmp24)
    tmp27 = triton_helpers.maximum(tmp25, tmp26)
    tmp29 = triton_helpers.maximum(tmp27, tmp28)
    tmp30 = triton_helpers.maximum(tmp22, tmp29)
    tl.store(out_ptr0 + (x0), tmp30, xmask)
''', device_str='cuda')


cpp_fused_copy_zeros_4 = async_compile.cpp_pybinding(['const float*', 'const float*', 'const float*', 'const float*', 'float*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(4L); x1+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp6 = in_ptr0[static_cast<int64_t>(x0)];
                        auto tmp9 = in_ptr1[static_cast<int64_t>(x0)];
                        auto tmp12 = in_ptr2[static_cast<int64_t>(x0)];
                        auto tmp14 = in_ptr3[static_cast<int64_t>(x0)];
                        auto tmp0 = x1;
                        auto tmp1 = c10::convert<int32_t>(tmp0);
                        auto tmp2 = static_cast<int32_t>(3);
                        auto tmp3 = tmp1 == tmp2;
                        auto tmp4 = static_cast<int32_t>(0);
                        auto tmp5 = tmp4 == tmp4;
                        auto tmp7 = static_cast<int32_t>(2);
                        auto tmp8 = tmp2 == tmp7;
                        auto tmp10 = static_cast<int32_t>(1);
                        auto tmp11 = tmp7 == tmp10;
                        auto tmp13 = tmp10 == tmp4;
                        auto tmp15 = static_cast<float>(0.0);
                        auto tmp16 = tmp5 ? tmp14 : tmp15;
                        auto tmp17 = tmp5 ? tmp16 : tmp15;
                        auto tmp18 = tmp13 ? tmp17 : tmp15;
                        auto tmp19 = tmp5 ? tmp12 : tmp18;
                        auto tmp20 = tmp5 ? tmp19 : tmp18;
                        auto tmp21 = tmp7 == tmp4;
                        auto tmp22 = tmp21 ? tmp17 : tmp15;
                        auto tmp23 = tmp11 ? tmp20 : tmp22;
                        auto tmp24 = tmp5 ? tmp9 : tmp23;
                        auto tmp25 = tmp5 ? tmp24 : tmp23;
                        auto tmp26 = tmp2 == tmp10;
                        auto tmp27 = tmp2 == tmp4;
                        auto tmp28 = tmp27 ? tmp17 : tmp15;
                        auto tmp29 = tmp26 ? tmp20 : tmp28;
                        auto tmp30 = tmp8 ? tmp25 : tmp29;
                        auto tmp31 = tmp5 ? tmp6 : tmp30;
                        auto tmp32 = tmp5 ? tmp31 : tmp30;
                        auto tmp33 = tmp1 == tmp7;
                        auto tmp34 = tmp1 == tmp10;
                        auto tmp35 = tmp1 == tmp4;
                        auto tmp36 = tmp35 ? tmp17 : tmp15;
                        auto tmp37 = tmp34 ? tmp20 : tmp36;
                        auto tmp38 = tmp33 ? tmp25 : tmp37;
                        auto tmp39 = tmp3 ? tmp32 : tmp38;
                        out_ptr0[static_cast<int64_t>(x1 + 4L*x0)] = tmp39;
                    }
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [max_1, max_2], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_0.run(arg0_1, buf0, 4, grid=grid(4), stream=stream0)
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [max_3, max_4], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_1.run(arg0_1, buf1, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [max_5, max_6], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_2.run(arg0_1, buf2, 4, grid=grid(4), stream=stream0)
        buf3 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [max_7, max_8], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(arg0_1, buf3, 4, grid=grid(4), stream=stream0)
        del arg0_1
    buf4 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf4.copy_(buf0, False)
    del buf0
    buf5 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf5.copy_(buf1, False)
    del buf1
    buf6 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf6.copy_(buf2, False)
    del buf2
    buf7 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf7.copy_(buf3, False)
    del buf3
    buf8 = empty_strided_cpu((4, 4, 1, 1), (4, 1, 1, 1), torch.float32)
    cpp_fused_copy_zeros_4(buf7, buf6, buf5, buf4, buf8)
    return (buf8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
