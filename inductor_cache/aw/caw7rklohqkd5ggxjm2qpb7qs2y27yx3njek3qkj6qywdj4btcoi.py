# AOT ID: ['13_forward']
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


# kernel path: inductor_cache/tl/ctlcmurnybol56rd3mydb7psvatofu7monmvjadbjijyk35j5sf7.py
# Topologically Sorted Source Nodes: [point_score], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   point_score => clone, clone_1
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 4*y1), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2 + 4*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3.to(tl.int64)
    tmp5 = y0
    tmp6 = tmp4 == tmp5
    tmp7 = tmp6.to(tl.int64)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8 * tmp1
    tl.store(out_ptr0 + (x2 + 4*y3), tmp2, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 4*y3), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5stecfxtslaxeflvn65biegkelc4nrn2npcbk62jl6tgotvt7ix.py
# Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   trans_score => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%unsqueeze_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 3)
    x2 = xindex // 12
    x0 = (xindex % 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = x0
    tmp3 = tmp1 == tmp2
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i3/ci3l3nt3nggwmsoozkdufoo4l27i27sj2uolpopa35r3ol57jvf3.py
# Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   trans_score => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_11,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 48
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 3)
    x2 = xindex // 12
    x1 = ((xindex // 3) % 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (1 + x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.int64)
    tmp2 = x1
    tmp3 = tmp1 == tmp2
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h5/ch5yawu37romlfao2r3pvpf37itjw5zxc47t64evxg5gwacbkstd.py
# Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   trans_score => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_12,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 12*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 3*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xo/cxoka53ucygxiyrqdjj5igxm2bsomp6oz6je55v36ghnxygjbyxg.py
# Topologically Sorted Source Nodes: [add_1, max_1, sub, exp, sum_1, outputs_1, mul_2, sub_1, mul_3, outputs_2], Original ATen: [aten.add, aten.max, aten.sub, aten.exp, aten.sum, aten.mul, aten.rsub]
# Source node to ATen node mapping:
#   add_1 => add_1
#   exp => exp
#   max_1 => max_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   outputs_1 => add_3
#   outputs_2 => add_4
#   sub => sub
#   sub_1 => sub_1
#   sum_1 => sum_1
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_5, %unsqueeze_6), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%add_1, 1, True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze, %select_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %add_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %select_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %squeeze_1), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %mul_3), kwargs = {})
triton_poi_fused_add_exp_max_mul_rsub_sub_sum_4 = async_compile.triton('triton_poi_fused_add_exp_max_mul_rsub_sub_sum_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_max_mul_rsub_sub_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_max_mul_rsub_sub_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 16*x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (2 + 16*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (3 + 16*x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr2 + (12 + x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp40 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp5 * tmp1
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp4, tmp8)
    tmp11 = tmp10 * tmp1
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(tmp9, tmp13)
    tmp16 = tmp15 * tmp1
    tmp18 = tmp16 + tmp17
    tmp19 = triton_helpers.maximum(tmp14, tmp18)
    tmp20 = tmp4 - tmp19
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp8 - tmp19
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp13 - tmp19
    tmp26 = tl_math.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp18 - tmp19
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp32 = tl_math.log(tmp30)
    tmp33 = tmp19 + tmp32
    tmp35 = tmp34 * tmp31
    tmp36 = tmp33 + tmp35
    tmp37 = tmp31 * tmp36
    tmp38 = 1.0
    tmp39 = tmp38 - tmp31
    tmp41 = tmp40 * tmp1
    tmp42 = tmp39 * tmp41
    tmp43 = tmp37 + tmp42
    tl.store(in_out_ptr0 + (x2), tmp43, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3x/c3xjajwo6ty5rmos2yqmfharaypog7lxdwl7jbdqraww533ksrpx.py
# Topologically Sorted Source Nodes: [add_5, max_2, sub_2, exp_1, sum_2, outputs_4, mul_4, sub_3, mul_5, outputs_5], Original ATen: [aten.add, aten.max, aten.sub, aten.exp, aten.sum, aten.mul, aten.rsub]
# Source node to ATen node mapping:
#   add_5 => add_5
#   exp_1 => exp_1
#   max_2 => max_2
#   mul_4 => mul_4
#   mul_5 => mul_5
#   outputs_4 => add_7
#   outputs_5 => add_8
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sum_2 => sum_2
# Graph fragment:
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_7, %unsqueeze_6), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%add_5, 1, True), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_2), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_2,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_1, [1], True), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_2, %select_4), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %add_7), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %select_3), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %squeeze_3), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %mul_5), kwargs = {})
triton_poi_fused_add_exp_max_mul_rsub_sub_sum_5 = async_compile.triton('triton_poi_fused_add_exp_max_mul_rsub_sub_sum_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_max_mul_rsub_sub_sum_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_max_mul_rsub_sub_sum_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (12 + x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (8 + x0 + 16*x1), xmask)
    tmp35 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = triton_helpers.maximum(tmp2, tmp5)
    tmp9 = tmp7 + tmp8
    tmp10 = triton_helpers.maximum(tmp6, tmp9)
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(tmp10, tmp13)
    tmp15 = tmp2 - tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp5 - tmp14
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp9 - tmp14
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp13 - tmp14
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tmp22 + tmp24
    tmp27 = tl_math.log(tmp25)
    tmp28 = tmp14 + tmp27
    tmp30 = tmp29 * tmp26
    tmp31 = tmp28 + tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = 1.0
    tmp34 = tmp33 - tmp26
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 + tmp36
    tl.store(in_out_ptr0 + (x2), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/el/celuen3mvnqtot2y6eyektmamgb6fc3gn3s4735ceflmborkttb6.py
# Topologically Sorted Source Nodes: [add_9, max_3, sub_4, exp_2, sum_3, outputs_7, mul_6, sub_5, mul_7, outputs_8], Original ATen: [aten.add, aten.max, aten.sub, aten.exp, aten.sum, aten.mul, aten.rsub]
# Source node to ATen node mapping:
#   add_9 => add_9
#   exp_2 => exp_2
#   max_3 => max_3
#   mul_6 => mul_6
#   mul_7 => mul_7
#   outputs_7 => add_11
#   outputs_8 => add_12
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sum_3 => sum_3
# Graph fragment:
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_9, %unsqueeze_6), kwargs = {})
#   %max_3 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%add_9, 1, True), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_4), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [1], True), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%squeeze_4, %select_6), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %add_11), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %select_5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %squeeze_5), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, %mul_7), kwargs = {})
triton_poi_fused_add_exp_max_mul_rsub_sub_sum_6 = async_compile.triton('triton_poi_fused_add_exp_max_mul_rsub_sub_sum_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_max_mul_rsub_sub_sum_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 11, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_max_mul_rsub_sub_sum_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (4 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (8 + x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (12 + x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr3 + (12 + x0 + 16*x1), xmask)
    tmp35 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = triton_helpers.maximum(tmp2, tmp5)
    tmp9 = tmp7 + tmp8
    tmp10 = triton_helpers.maximum(tmp6, tmp9)
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(tmp10, tmp13)
    tmp15 = tmp2 - tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp5 - tmp14
    tmp18 = tl_math.exp(tmp17)
    tmp19 = tmp16 + tmp18
    tmp20 = tmp9 - tmp14
    tmp21 = tl_math.exp(tmp20)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp13 - tmp14
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tmp22 + tmp24
    tmp27 = tl_math.log(tmp25)
    tmp28 = tmp14 + tmp27
    tmp30 = tmp29 * tmp26
    tmp31 = tmp28 + tmp30
    tmp32 = tmp26 * tmp31
    tmp33 = 1.0
    tmp34 = tmp33 - tmp26
    tmp36 = tmp34 * tmp35
    tmp37 = tmp32 + tmp36
    tl.store(in_out_ptr0 + (x2), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ft/cfteintfq23bhrsyae4d5kbkclqwi3abc5akg7ypowibmpo5iww7.py
# Topologically Sorted Source Nodes: [target_score, sub_7], Original ATen: [aten.add, aten.sub]
# Source node to ATen node mapping:
#   sub_7 => sub_7
#   target_score => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_5, %view_12), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%squeeze_6, %add), kwargs = {})
triton_poi_fused_add_sub_7 = async_compile.triton('triton_poi_fused_add_sub_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sub_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sub_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp21 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = triton_helpers.maximum(tmp0, tmp1)
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = tmp0 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp9 = tmp1 - tmp6
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp8 + tmp10
    tmp12 = tmp3 - tmp6
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tmp11 + tmp13
    tmp15 = tmp5 - tmp6
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp14 + tmp16
    tmp18 = tl_math.log(tmp17)
    tmp19 = tmp6 + tmp18
    tmp22 = tmp20 + tmp21
    tmp23 = tmp19 - tmp22
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 1), (4, 1, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [point_score], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(primals_1, primals_2, primals_3, buf0, buf1, 16, 4, grid=grid(16, 4), stream=stream0)
        buf2 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [point_score], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf0, (4, 1, 16), (16, 0, 1), 0), reinterpret_tensor(buf1, (4, 16, 1), (16, 1, 0), 0), out=buf2)
        del buf0
        del buf1
        buf3 = empty_strided_cuda((4, 3, 4, 1), (12, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(primals_3, primals_2, buf3, 48, grid=grid(48), stream=stream0)
        buf4 = empty_strided_cuda((1, 12, 4), (48, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (1, 12, 4), (0, 4, 1), 0), reinterpret_tensor(primals_4, (1, 4, 4), (16, 4, 1), 0), out=buf4)
        buf5 = empty_strided_cuda((4, 4, 3, 1), (12, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(primals_3, primals_2, buf5, 48, grid=grid(48), stream=stream0)
        del primals_3
        buf6 = empty_strided_cuda((4, 4, 3, 1), (12, 3, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf4, buf6, 16, 3, grid=grid(16, 3), stream=stream0)
        del buf4
        buf7 = empty_strided_cuda((4, 1, 1), (1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [trans_score], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf5, (4, 1, 12), (12, 0, 1), 0), reinterpret_tensor(buf6, (4, 12, 1), (12, 1, 0), 0), out=buf7)
        del buf6
        buf8 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        buf10 = reinterpret_tensor(buf8, (4, 4), (4, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [add_1, max_1, sub, exp, sum_1, outputs_1, mul_2, sub_1, mul_3, outputs_2], Original ATen: [aten.add, aten.max, aten.sub, aten.exp, aten.sum, aten.mul, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_max_mul_rsub_sub_sum_4.run(buf10, primals_1, primals_2, primals_4, 16, grid=grid(16), stream=stream0)
        buf11 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        buf13 = reinterpret_tensor(buf11, (4, 4), (4, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [add_5, max_2, sub_2, exp_1, sum_2, outputs_4, mul_4, sub_3, mul_5, outputs_5], Original ATen: [aten.add, aten.max, aten.sub, aten.exp, aten.sum, aten.mul, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_max_mul_rsub_sub_sum_5.run(buf13, buf10, primals_4, primals_2, primals_1, 16, grid=grid(16), stream=stream0)
        buf14 = reinterpret_tensor(buf10, (4, 1, 4), (4, 16, 1), 0); del buf10  # reuse
        buf16 = reinterpret_tensor(buf14, (4, 4), (4, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [add_9, max_3, sub_4, exp_2, sum_3, outputs_7, mul_6, sub_5, mul_7, outputs_8], Original ATen: [aten.add, aten.max, aten.sub, aten.exp, aten.sum, aten.mul, aten.rsub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_max_mul_rsub_sub_sum_6.run(buf16, buf13, primals_4, primals_2, primals_1, 16, grid=grid(16), stream=stream0)
        del buf13
        buf17 = reinterpret_tensor(buf2, (4, ), (1, ), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [target_score, sub_7], Original ATen: [aten.add, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sub_7.run(buf17, buf16, buf7, 4, grid=grid(4), stream=stream0)
        del buf16
        del buf7
    return (buf17, primals_1, primals_2, primals_4, reinterpret_tensor(buf5, (4, 12, 1), (12, 1, 12), 0), reinterpret_tensor(buf3, (1, 4, 12), (48, 1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
