# AOT ID: ['15_forward']
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


# kernel path: inductor_cache/z2/cz2ln3hq3tlz34bqgu22tuqa6dp4bwedne32we76oebob6cl465u.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   input_2 => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_1,), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_0 = async_compile.triton('triton_poi_fused_relu_threshold_backward_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l3/cl3q6t4lp7xwyydhr3memgpt6qeu56bqzwwbjstjs5jp3swagtgf.py
# Topologically Sorted Source Nodes: [anchor], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   anchor => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %view_3), kwargs = {})
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ij/cij4pd7ouabh6qlhhoocrsct6s43jzrdxlab3f3wdkn5i4oycoja.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits => mm_1
# Graph fragment:
#   %mm_1 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_3), kwargs = {})
triton_poi_fused_mm_2 = async_compile.triton('triton_poi_fused_mm_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ef/cefsabjkzlmchry2axoswz3772ekrbwctfj7hdmnkgusmk4v6vlh.py
# Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
# Source node to ATen node mapping:
#   max_1 => getitem_1
# Graph fragment:
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%max_1, 1), kwargs = {})
triton_poi_fused_max_3 = async_compile.triton('triton_poi_fused_max_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp17 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp32 = tl.load(in_ptr0 + (48 + x0 + 64*x1), xmask)
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
    tl.store(out_ptr0 + (x2), tmp46, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdo7f4an7hcogjrjxl4zhqmapisanwhcmv3kuqvuqpk4wo3tb7oa.py
# Topologically Sorted Source Nodes: [max_1, logits_1], Original ATen: [aten.max, aten.sub]
# Source node to ATen node mapping:
#   logits_1 => sub
#   max_1 => max_1
# Graph fragment:
#   %max_1 : [num_users=2] = call_function[target=torch.ops.aten.max.dim](args = (%view_7, 1, True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_7, %getitem), kwargs = {})
triton_poi_fused_max_sub_4 = async_compile.triton('triton_poi_fused_max_sub_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_sub_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_sub_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xo/cxo2g7ryixvw4dgwmozhee2lj753srtpciovhnit2ixlfodzdzjj.py
# Topologically Sorted Source Nodes: [logits_2], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_2 => mm_2
# Graph fragment:
#   %mm_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_4), kwargs = {})
triton_poi_fused_mm_5 = async_compile.triton('triton_poi_fused_mm_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g4/cg4vj53ezlrszfzqx3zn2amufnqnyzaqwyy263p4rsa7x5k7zhnd.py
# Topologically Sorted Source Nodes: [logits_4], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_4 => mm_3
# Graph fragment:
#   %mm_3 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_5), kwargs = {})
triton_poi_fused_mm_6 = async_compile.triton('triton_poi_fused_mm_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gs/cgsjb622ygae6wdq2vmhngkgjxpk3npcsblqc35dd3tvruvac6gd.py
# Topologically Sorted Source Nodes: [logits_6], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_6 => mm_4
# Graph fragment:
#   %mm_4 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_6), kwargs = {})
triton_poi_fused_mm_7 = async_compile.triton('triton_poi_fused_mm_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d4/cd457jb6hc2blqrwfnkckyh222it4kdlumgiqjiy6ufqfnds6mpo.py
# Topologically Sorted Source Nodes: [logits_8], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_8 => mm_5
# Graph fragment:
#   %mm_5 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_7), kwargs = {})
triton_poi_fused_mm_8 = async_compile.triton('triton_poi_fused_mm_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nf/cnfeeuetuaoz26aodosgucnljlcvui7hn2icv4wrle7snemlhjxo.py
# Topologically Sorted Source Nodes: [logits_10], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_10 => mm_6
# Graph fragment:
#   %mm_6 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_8), kwargs = {})
triton_poi_fused_mm_9 = async_compile.triton('triton_poi_fused_mm_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3h/c3hj3c6jiebycyhccmazqx6ww6c6nkqxt4kx3xlvkpv7fmhzmok2.py
# Topologically Sorted Source Nodes: [logits_12], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_12 => mm_7
# Graph fragment:
#   %mm_7 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_9), kwargs = {})
triton_poi_fused_mm_10 = async_compile.triton('triton_poi_fused_mm_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/td/ctd2ngt4oozw5oxoeoxptactps4cv25nukqkyx7fgys23gezvyhw.py
# Topologically Sorted Source Nodes: [logits_14], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_14 => mm_8
# Graph fragment:
#   %mm_8 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_10), kwargs = {})
triton_poi_fused_mm_11 = async_compile.triton('triton_poi_fused_mm_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ic/cicbcslglqy4wbtgbur5lvcgnvcyvgafidg7rzzbmlsh3lpunrvw.py
# Topologically Sorted Source Nodes: [logits_16], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_16 => mm_9
# Graph fragment:
#   %mm_9 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_11), kwargs = {})
triton_poi_fused_mm_12 = async_compile.triton('triton_poi_fused_mm_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7w/c7wsvfgoc4ytwtso4rh6nvysza4s6ul6y52heycasmdmnh3eynqy.py
# Topologically Sorted Source Nodes: [logits_18], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_18 => mm_10
# Graph fragment:
#   %mm_10 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_12), kwargs = {})
triton_poi_fused_mm_13 = async_compile.triton('triton_poi_fused_mm_13', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uz/cuz6e7kh46rhhkzwhcgkiypliqokjn62a5lwkw4h4wy3fkvss57v.py
# Topologically Sorted Source Nodes: [logits_20], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_20 => mm_11
# Graph fragment:
#   %mm_11 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_13), kwargs = {})
triton_poi_fused_mm_14 = async_compile.triton('triton_poi_fused_mm_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/um/cumuz633hhwnxjc75r7cinbpzfw2r67fjxcdfzs5cf5tg4kytmau.py
# Topologically Sorted Source Nodes: [logits_22], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_22 => mm_12
# Graph fragment:
#   %mm_12 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_14), kwargs = {})
triton_poi_fused_mm_15 = async_compile.triton('triton_poi_fused_mm_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/w6/cw6yz4pd5r23uee2kyivpvwo3x42odahbeswfa45hq3fzvp7mxdd.py
# Topologically Sorted Source Nodes: [logits_24], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_24 => mm_13
# Graph fragment:
#   %mm_13 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_15), kwargs = {})
triton_poi_fused_mm_16 = async_compile.triton('triton_poi_fused_mm_16', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ky/ckythrzsrl5lew5l3uaplebm4zpp6rvwnngtis42enx4yt2i3j36.py
# Topologically Sorted Source Nodes: [logits_26], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_26 => mm_14
# Graph fragment:
#   %mm_14 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_16), kwargs = {})
triton_poi_fused_mm_17 = async_compile.triton('triton_poi_fused_mm_17', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bg/cbgbewdgtdddmdnlye3icpw4qhtbrpqnfudezzxwkxx26e4la7j2.py
# Topologically Sorted Source Nodes: [logits_28], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_28 => mm_15
# Graph fragment:
#   %mm_15 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_17), kwargs = {})
triton_poi_fused_mm_18 = async_compile.triton('triton_poi_fused_mm_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cz/cczpvg4ui6gg5v7pp3nkz5be2eys77aprtaf7tde5ijdsuqle5hw.py
# Topologically Sorted Source Nodes: [logits_30], Original ATen: [aten.mm]
# Source node to ATen node mapping:
#   logits_30 => mm_16
# Graph fragment:
#   %mm_16 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_6, %permute_18), kwargs = {})
triton_poi_fused_mm_19 = async_compile.triton('triton_poi_fused_mm_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mm_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mm_19(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_7, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = reinterpret_tensor(buf0, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf0  # reuse
        buf69 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_0.run(buf1, primals_2, buf69, 256, grid=grid(256), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1, (64, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf2)
        buf3 = reinterpret_tensor(buf2, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [anchor], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf3, primals_3, primals_5, 256, grid=grid(256), stream=stream0)
        del primals_5
        buf4 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pred], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_7, (4, 4), (1, 4), 0), out=buf4)
        buf5 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_2.run(primals_6, buf5, 16, grid=grid(16), stream=stream0)
        buf6 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf5, out=buf6)
        buf7 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_1], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf6, buf7, 64, grid=grid(64), stream=stream0)
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_1, logits_1], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf6, buf8, 256, grid=grid(256), stream=stream0)
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [logits_2], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_5.run(primals_6, buf9, 16, grid=grid(16), stream=stream0)
        buf10 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [logits_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf9, out=buf10)
        buf11 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_2], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf10, buf11, 64, grid=grid(64), stream=stream0)
        buf12 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_2, logits_3], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf10, buf12, 256, grid=grid(256), stream=stream0)
        buf13 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [logits_4], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_6.run(primals_6, buf13, 16, grid=grid(16), stream=stream0)
        buf14 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [logits_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf13, out=buf14)
        buf15 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_3], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf14, buf15, 64, grid=grid(64), stream=stream0)
        buf16 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_3, logits_5], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf14, buf16, 256, grid=grid(256), stream=stream0)
        buf17 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [logits_6], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_7.run(primals_6, buf17, 16, grid=grid(16), stream=stream0)
        buf18 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [logits_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf17, out=buf18)
        buf19 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_4], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf18, buf19, 64, grid=grid(64), stream=stream0)
        buf20 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_4, logits_7], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf18, buf20, 256, grid=grid(256), stream=stream0)
        buf21 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [logits_8], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_8.run(primals_6, buf21, 16, grid=grid(16), stream=stream0)
        buf22 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [logits_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf21, out=buf22)
        buf23 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_5], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf22, buf23, 64, grid=grid(64), stream=stream0)
        buf24 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_5, logits_9], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf22, buf24, 256, grid=grid(256), stream=stream0)
        buf25 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [logits_10], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_9.run(primals_6, buf25, 16, grid=grid(16), stream=stream0)
        buf26 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [logits_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf25, out=buf26)
        buf27 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_6], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf26, buf27, 64, grid=grid(64), stream=stream0)
        buf28 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_6, logits_11], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf26, buf28, 256, grid=grid(256), stream=stream0)
        buf29 = buf25; del buf25  # reuse
        # Topologically Sorted Source Nodes: [logits_12], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_10.run(primals_6, buf29, 16, grid=grid(16), stream=stream0)
        buf30 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [logits_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf29, out=buf30)
        buf31 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_7], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf30, buf31, 64, grid=grid(64), stream=stream0)
        buf32 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_7, logits_13], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf30, buf32, 256, grid=grid(256), stream=stream0)
        buf33 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [logits_14], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_11.run(primals_6, buf33, 16, grid=grid(16), stream=stream0)
        buf34 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [logits_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf33, out=buf34)
        buf35 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_8], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf34, buf35, 64, grid=grid(64), stream=stream0)
        buf36 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_8, logits_15], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf34, buf36, 256, grid=grid(256), stream=stream0)
        buf37 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [logits_16], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_12.run(primals_6, buf37, 16, grid=grid(16), stream=stream0)
        buf38 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [logits_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf37, out=buf38)
        buf39 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_9], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf38, buf39, 64, grid=grid(64), stream=stream0)
        buf40 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_9, logits_17], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf38, buf40, 256, grid=grid(256), stream=stream0)
        buf41 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [logits_18], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_13.run(primals_6, buf41, 16, grid=grid(16), stream=stream0)
        buf42 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [logits_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf41, out=buf42)
        buf43 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_10], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf42, buf43, 64, grid=grid(64), stream=stream0)
        buf44 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_10, logits_19], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf42, buf44, 256, grid=grid(256), stream=stream0)
        buf45 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [logits_20], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_14.run(primals_6, buf45, 16, grid=grid(16), stream=stream0)
        buf46 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [logits_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf45, out=buf46)
        buf47 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_11], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf46, buf47, 64, grid=grid(64), stream=stream0)
        buf48 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_11, logits_21], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf46, buf48, 256, grid=grid(256), stream=stream0)
        buf49 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [logits_22], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_15.run(primals_6, buf49, 16, grid=grid(16), stream=stream0)
        buf50 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [logits_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf49, out=buf50)
        buf51 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_12], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf50, buf51, 64, grid=grid(64), stream=stream0)
        buf52 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_12, logits_23], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf50, buf52, 256, grid=grid(256), stream=stream0)
        buf53 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [logits_24], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_16.run(primals_6, buf53, 16, grid=grid(16), stream=stream0)
        buf54 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [logits_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf53, out=buf54)
        buf55 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_13], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf54, buf55, 64, grid=grid(64), stream=stream0)
        buf56 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_13, logits_25], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf54, buf56, 256, grid=grid(256), stream=stream0)
        buf57 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [logits_26], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_17.run(primals_6, buf57, 16, grid=grid(16), stream=stream0)
        buf58 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [logits_26], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf57, out=buf58)
        buf59 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_14], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf58, buf59, 64, grid=grid(64), stream=stream0)
        buf60 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_14, logits_27], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf58, buf60, 256, grid=grid(256), stream=stream0)
        buf61 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [logits_28], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_18.run(primals_6, buf61, 16, grid=grid(16), stream=stream0)
        buf62 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [logits_28], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf61, out=buf62)
        buf63 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_15], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf62, buf63, 64, grid=grid(64), stream=stream0)
        buf64 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [max_15, logits_29], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf62, buf64, 256, grid=grid(256), stream=stream0)
        buf65 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [logits_30], Original ATen: [aten.mm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mm_19.run(primals_6, buf65, 16, grid=grid(16), stream=stream0)
        buf66 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [logits_30], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf65, out=buf66)
        del buf65
        buf67 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.int64)
        # Topologically Sorted Source Nodes: [max_16], Original ATen: [aten.max]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_3.run(buf66, buf67, 64, grid=grid(64), stream=stream0)
        buf68 = reinterpret_tensor(buf4, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [max_16, logits_31], Original ATen: [aten.max, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_sub_4.run(buf66, buf68, 256, grid=grid(256), stream=stream0)
        del buf66
    return (buf8, buf12, buf16, buf20, buf24, buf28, buf32, buf36, buf40, buf44, buf48, buf52, buf56, buf60, buf64, buf68, reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(buf1, (64, 4), (4, 1), 0), reinterpret_tensor(buf3, (64, 4), (4, 1), 0), buf7, buf11, buf15, buf19, buf23, buf27, buf31, buf35, buf39, buf43, buf47, buf51, buf55, buf59, buf63, buf67, reinterpret_tensor(primals_6, (4, 4), (64, 16), 15), reinterpret_tensor(primals_6, (4, 4), (64, 16), 14), reinterpret_tensor(primals_6, (4, 4), (64, 16), 13), reinterpret_tensor(primals_6, (4, 4), (64, 16), 12), reinterpret_tensor(primals_6, (4, 4), (64, 16), 11), reinterpret_tensor(primals_6, (4, 4), (64, 16), 10), reinterpret_tensor(primals_6, (4, 4), (64, 16), 9), reinterpret_tensor(primals_6, (4, 4), (64, 16), 8), reinterpret_tensor(primals_6, (4, 4), (64, 16), 7), reinterpret_tensor(primals_6, (4, 4), (64, 16), 6), reinterpret_tensor(primals_6, (4, 4), (64, 16), 5), reinterpret_tensor(primals_6, (4, 4), (64, 16), 4), reinterpret_tensor(primals_6, (4, 4), (64, 16), 3), reinterpret_tensor(primals_6, (4, 4), (64, 16), 2), reinterpret_tensor(primals_6, (4, 4), (64, 16), 1), reinterpret_tensor(primals_6, (4, 4), (64, 16), 0), primals_7, primals_4, buf69, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
