# AOT ID: ['23_forward']
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


# kernel path: inductor_cache/qy/cqyn6hnk2falnyqt423mlcysxgf5653crqkzjpfj5mvt6g5mlbdu.py
# Topologically Sorted Source Nodes: [repeat], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   repeat => repeat
# Graph fragment:
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_1, [8, 1, 1]), kwargs = {})
triton_poi_fused_repeat_0 = async_compile.triton('triton_poi_fused_repeat_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*((x1 % 4))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bv/cbvuitmbazao6fxw2aw6ecmlpryeq7jy4tem67t64icuuddmvhw7.py
# Topologically Sorted Source Nodes: [bmm, bmm_1, bmm_2], Original ATen: [aten.bmm, aten.transpose]
# Source node to ATen node mapping:
#   bmm => bmm
#   bmm_1 => bmm_1
#   bmm_2 => bmm_2
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %primals_2), kwargs = {})
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %primals_3), kwargs = {})
#   %bmm_2 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %primals_4), kwargs = {})
#   %permute_30 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
triton_poi_fused_bmm_transpose_1 = async_compile.triton('triton_poi_fused_bmm_transpose_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_transpose_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_transpose_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 16)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*((x1 % 4)) + 16*((x0 + 4*x1) // 16) + 64*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
    tl.store(out_ptr1 + (x3), tmp0, xmask)
    tl.store(out_ptr2 + (x3), tmp0, xmask)
    tl.store(out_ptr3 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jx/cjxtq2qwcbs22eczrwtd2mmrqitdo5462vqu3k3tfdtty3v2vn5v.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   out => amax, exp, sub, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_6, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_6, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_poi_fused__softmax_2 = async_compile.triton('triton_poi_fused__softmax_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = in_ptr1
    tmp4 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp4 / tmp2
    tmp6 = triton_helpers.maximum(tmp3, tmp5)
    tmp8 = tmp7 / tmp2
    tmp9 = triton_helpers.maximum(tmp6, tmp8)
    tmp11 = tmp10 / tmp2
    tmp12 = triton_helpers.maximum(tmp9, tmp11)
    tmp13 = tmp3 - tmp12
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp5 - tmp12
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp14 + tmp16
    tmp18 = tmp8 - tmp12
    tmp19 = tl_math.exp(tmp18)
    tmp20 = tmp17 + tmp19
    tmp21 = tmp11 - tmp12
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp20 + tmp22
    tl.store(out_ptr0 + (x0), tmp12, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rc/crcdfg6yp2jl2mgbgaj5gasvjprcsg4qkqwud4k4znrevn6bjss4.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   out => amax, div_1, exp, sub, sum_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_6, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_6, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_3 = async_compile.triton('triton_poi_fused__softmax_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': 'fp64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = in_ptr0
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 / tmp2
    tmp5 = tmp3 - tmp4
    tmp6 = tl_math.exp(tmp5)
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7n/c7nq62jrcev3h2by2zjkeu3yn6xkv2ww3jvj7haigkjzyyzgrvse.py
# Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %getitem_1, %getitem_2, %getitem_3, %getitem_4, %getitem_5, %getitem_6, %getitem_7], -1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x1 = xindex // 32
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (64 + 4*x1 + ((-4) + x0)), tmp9 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr0 + (128 + 4*x1 + ((-8) + x0)), tmp14 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 16, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr0 + (192 + 4*x1 + ((-12) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 20, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr0 + (256 + 4*x1 + ((-16) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 24, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr0 + (320 + 4*x1 + ((-20) + x0)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 28, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr0 + (384 + 4*x1 + ((-24) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 32, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr0 + (448 + 4*x1 + ((-28) + x0)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp14, tmp15, tmp43)
    tmp45 = tl.where(tmp9, tmp10, tmp44)
    tmp46 = tl.where(tmp4, tmp5, tmp45)
    tl.store(out_ptr0 + (x2), tmp46, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pe/cpeknlmavmmykl7rwx27fz2figrghwcu7l3lp6ilrqr3hcprpmri.py
# Topologically Sorted Source Nodes: [outputs_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   outputs_3 => add
# Graph fragment:
#   %add : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_9, %primals_1), kwargs = {})
triton_poi_fused_add_5 = async_compile.triton('triton_poi_fused_add_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m3/cm3uiaib2lf6khslsqzwy7n5vikk25ih6nqyex2h5qiv7x6cazwo.py
# Topologically Sorted Source Nodes: [sub, add_1, ln_out, mul, ln_out_1], Original ATen: [aten.sub, aten.add, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_1
#   ln_out => div_2
#   ln_out_1 => add_2
#   mul => mul
#   sub => sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %expand), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_1, 0.001), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_1, %add_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %expand_2), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %expand_3), kwargs = {})
triton_poi_fused_add_div_mul_sub_6 = async_compile.triton('triton_poi_fused_add_div_mul_sub_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sub_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 4.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp0 - tmp9
    tmp11 = tmp1 - tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp2 - tmp9
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp16 = tmp4 - tmp9
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = tmp6 - tmp9
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = 3.0
    tmp23 = tmp21 / tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = 0.001
    tmp26 = tmp24 + tmp25
    tmp27 = tmp10 / tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4m/c4mq2etlf6cc5rokkvtjgl45osnqz7da3hn5lwuuzvne5efivwzh.py
# Topologically Sorted Source Nodes: [bmm_5], Original ATen: [aten.bmm, aten.transpose]
# Source node to ATen node mapping:
#   bmm_5 => bmm_5
# Graph fragment:
#   %bmm_5 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_10, %primals_11), kwargs = {})
#   %permute_19 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_10, [0, 2, 1]), kwargs = {})
triton_poi_fused_bmm_transpose_7 = async_compile.triton('triton_poi_fused_bmm_transpose_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_transpose_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_transpose_7(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 16)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*((x1 % 4)) + 16*((x0 + 4*x1) // 16) + 64*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
    tl.store(out_ptr1 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zd/czdswdxlrx3dxqbh4svykktmohdrvpsleicbhveepkcudb5re4qy.py
# Topologically Sorted Source Nodes: [bmm_6, bmm_7], Original ATen: [aten.bmm, aten.transpose]
# Source node to ATen node mapping:
#   bmm_6 => bmm_6
#   bmm_7 => bmm_7
# Graph fragment:
#   %bmm_6 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_11, %primals_12), kwargs = {})
#   %bmm_7 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_11, %primals_13), kwargs = {})
#   %permute_17 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_11, [0, 2, 1]), kwargs = {})
triton_poi_fused_bmm_transpose_8 = async_compile.triton('triton_poi_fused_bmm_transpose_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_transpose_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_transpose_8(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 16)
    x2 = xindex // 64
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*((x1 % 4)) + 16*((x0 + 4*x1) // 16) + 64*x2), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
    tl.store(out_ptr1 + (x3), tmp0, xmask)
    tl.store(out_ptr2 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lk/clkrdubbnxbznp7gi5r3bk5uxwfqofmrolqr2br7iphd65ixlycp.py
# Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   conv1d => convolution
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %primals_19, %primals_20, [1], [0], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_9 = async_compile.triton('triton_poi_fused_convolution_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jp/cjpedjetiefk7puobyjekmiydvu5cpjo3t4ezuw24lqbnkg7cb73.py
# Topologically Sorted Source Nodes: [conv1d, output_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   conv1d => convolution
#   output_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %primals_19, %primals_20, [1], [0], [1], False, [0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_10 = async_compile.triton('triton_poi_fused_convolution_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6u/c6uu4rfbgo3xsxmnmfrre5taemc3o75ovm367clfkokiuvfkdncj.py
# Topologically Sorted Source Nodes: [output_5], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   output_5 => add_6
# Graph fragment:
#   %add_6 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_5, %add_5), kwargs = {})
triton_poi_fused_add_11 = async_compile.triton('triton_poi_fused_add_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_11(in_out_ptr0, in_ptr0, in_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/si/csitawp7zjkcovqaxgcwfflfxbkrkxzo5j5cpqtd2ujcwsznwkxp.py
# Topologically Sorted Source Nodes: [sub_2, add_7, ln_out_4, mul_2, ln_out_5], Original ATen: [aten.sub, aten.add, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add_7 => add_7
#   ln_out_4 => div_6
#   ln_out_5 => add_8
#   mul_2 => mul_2
#   sub_2 => sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %expand_8), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%expand_9, 0.001), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_4, %add_7), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_6, %expand_10), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %expand_11), kwargs = {})
triton_poi_fused_add_div_mul_sub_12 = async_compile.triton('triton_poi_fused_add_div_mul_sub_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mul_sub_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mul_sub_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    x1 = ((xindex // 4) % 4)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (4 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (8 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (12 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = 4.0
    tmp9 = tmp7 / tmp8
    tmp10 = tmp0 - tmp9
    tmp11 = tmp1 - tmp9
    tmp12 = tmp11 * tmp11
    tmp13 = tmp2 - tmp9
    tmp14 = tmp13 * tmp13
    tmp15 = tmp12 + tmp14
    tmp16 = tmp4 - tmp9
    tmp17 = tmp16 * tmp16
    tmp18 = tmp15 + tmp17
    tmp19 = tmp6 - tmp9
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = 3.0
    tmp23 = tmp21 / tmp22
    tmp24 = libdevice.sqrt(tmp23)
    tmp25 = 0.001
    tmp26 = tmp24 + tmp25
    tmp27 = tmp10 / tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr0 + (x3), tmp31, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (8, 4, 4), (16, 4, 1))
    assert_size_stride(primals_3, (8, 4, 4), (16, 4, 1))
    assert_size_stride(primals_4, (8, 4, 4), (16, 4, 1))
    assert_size_stride(primals_5, (), ())
    assert_size_stride(primals_6, (4, 32), (32, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_11, (8, 4, 4), (16, 4, 1))
    assert_size_stride(primals_12, (8, 4, 4), (16, 4, 1))
    assert_size_stride(primals_13, (8, 4, 4), (16, 4, 1))
    assert_size_stride(primals_14, (), ())
    assert_size_stride(primals_15, (4, 32), (32, 1))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, ), (1, ))
    assert_size_stride(primals_19, (16, 4, 1), (4, 1, 1))
    assert_size_stride(primals_20, (16, ), (1, ))
    assert_size_stride(primals_21, (4, 16, 1), (16, 1, 1))
    assert_size_stride(primals_22, (4, ), (1, ))
    assert_size_stride(primals_23, (4, ), (1, ))
    assert_size_stride(primals_24, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_0.run(primals_1, buf0, 512, grid=grid(512), stream=stream0)
        buf1 = empty_strided_cuda((8, 16, 4), (64, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((8, 16, 4), (64, 4, 1), torch.float32)
        buf5 = empty_strided_cuda((8, 16, 4), (64, 4, 1), torch.float32)
        buf41 = empty_strided_cuda((8, 4, 16), (64, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [bmm, bmm_1, bmm_2], Original ATen: [aten.bmm, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_transpose_1.run(buf0, buf1, buf3, buf5, buf41, 512, grid=grid(512), stream=stream0)
        buf2 = reinterpret_tensor(buf0, (8, 16, 4), (64, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf1, primals_2, out=buf2)
        del primals_2
        buf4 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf3, primals_3, out=buf4)
        del primals_3
        buf6 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [bmm_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf5, primals_4, out=buf6)
        del primals_4
        buf7 = reinterpret_tensor(buf5, (32, 4, 4), (16, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (32, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf4, (32, 4, 4), (16, 1, 4), 0), out=buf7)
        buf8 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        buf9 = empty_strided_cuda((128, 1), (1, 128), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf7, primals_5.item(), buf8, buf9, 128, grid=grid(128), stream=stream0)
        buf10 = reinterpret_tensor(buf7, (128, 4), (4, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf10, primals_5.item(), buf8, buf9, 512, grid=grid(512), stream=stream0)
        buf11 = empty_strided_cuda((32, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (32, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf6, (32, 4, 4), (16, 4, 1), 0), out=buf11)
        buf12 = empty_strided_cuda((4, 4, 32), (128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf11, buf12, 512, grid=grid(512), stream=stream0)
        buf13 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf12, (16, 32), (32, 1), 0), reinterpret_tensor(primals_6, (32, 4), (1, 32), 0), out=buf13)
        buf14 = reinterpret_tensor(buf13, (4, 4, 4), (16, 4, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [outputs_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf14, primals_7, primals_1, 64, grid=grid(64), stream=stream0)
        del primals_1
        del primals_7
        buf15 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sub, add_1, ln_out, mul, ln_out_1], Original ATen: [aten.sub, aten.add, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_6.run(buf14, primals_8, primals_9, buf15, 64, grid=grid(64), stream=stream0)
        del primals_9
        buf16 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [repeat_3], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_0.run(buf15, buf16, 512, grid=grid(512), stream=stream0)
        buf17 = empty_strided_cuda((8, 16, 4), (64, 4, 1), torch.float32)
        buf40 = empty_strided_cuda((8, 4, 16), (64, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_5], Original ATen: [aten.bmm, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_transpose_7.run(buf16, buf17, buf40, 512, grid=grid(512), stream=stream0)
        buf18 = reinterpret_tensor(buf16, (8, 16, 4), (64, 4, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [bmm_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf17, primals_11, out=buf18)
        buf19 = reinterpret_tensor(buf17, (32, 4, 4), (16, 4, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [repeat_4], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_0.run(primals_10, buf19, 512, grid=grid(512), stream=stream0)
        del primals_10
        buf20 = empty_strided_cuda((8, 16, 4), (64, 4, 1), torch.float32)
        buf22 = empty_strided_cuda((8, 16, 4), (64, 4, 1), torch.float32)
        buf39 = empty_strided_cuda((8, 4, 16), (64, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [bmm_6, bmm_7], Original ATen: [aten.bmm, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_transpose_8.run(buf19, buf20, buf22, buf39, 512, grid=grid(512), stream=stream0)
        buf21 = reinterpret_tensor(buf19, (8, 16, 4), (64, 4, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [bmm_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf20, primals_12, out=buf21)
        del primals_12
        buf23 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [bmm_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf22, primals_13, out=buf23)
        del primals_13
        buf24 = reinterpret_tensor(buf22, (32, 4, 4), (16, 4, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [bmm_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (32, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf21, (32, 4, 4), (16, 1, 4), 0), out=buf24)
        buf25 = buf9; del buf9  # reuse
        buf26 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf24, primals_14.item(), buf25, buf26, 128, grid=grid(128), stream=stream0)
        buf27 = reinterpret_tensor(buf24, (128, 4), (4, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [out_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf27, primals_14.item(), buf25, buf26, 512, grid=grid(512), stream=stream0)
        del buf25
        del buf26
        buf28 = empty_strided_cuda((32, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (32, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf23, (32, 4, 4), (16, 4, 1), 0), out=buf28)
        buf29 = empty_strided_cuda((4, 4, 32), (128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf28, buf29, 512, grid=grid(512), stream=stream0)
        del buf28
        buf30 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf29, (16, 32), (32, 1), 0), reinterpret_tensor(primals_15, (32, 4), (1, 32), 0), out=buf30)
        buf31 = reinterpret_tensor(buf30, (4, 4, 4), (16, 4, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [outputs_7], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf31, primals_16, buf15, 64, grid=grid(64), stream=stream0)
        del primals_16
        buf32 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [sub_1, add_4, ln_out_2, mul_1, ln_out_3], Original ATen: [aten.sub, aten.add, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_6.run(buf31, primals_17, primals_18, buf32, 64, grid=grid(64), stream=stream0)
        del primals_18
        buf33 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_9.run(buf32, buf33, 16, 4, grid=grid(16, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [conv1d], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_19, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf34, (4, 16, 4), (64, 4, 1))
        buf35 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [conv1d, output_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_10.run(buf35, primals_20, 256, grid=grid(256), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [conv1d_1], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_21, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf36, (4, 4, 4), (16, 4, 1))
        buf37 = reinterpret_tensor(buf36, (4, 4, 4), (16, 1, 4), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [output_5], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_11.run(buf37, primals_22, buf32, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_22
        buf38 = reinterpret_tensor(buf33, (4, 4, 4), (16, 1, 4), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [sub_2, add_7, ln_out_4, mul_2, ln_out_5], Original ATen: [aten.sub, aten.add, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_12.run(buf37, primals_23, primals_24, buf38, 64, grid=grid(64), stream=stream0)
        del primals_24
    return (buf38, primals_5, primals_8, primals_14, primals_17, primals_19, primals_21, primals_23, buf10, reinterpret_tensor(buf12, (16, 32), (32, 1), 0), buf14, buf27, reinterpret_tensor(buf29, (16, 32), (32, 1), 0), buf31, reinterpret_tensor(buf32, (4, 4, 4), (16, 1, 4), 0), buf35, buf37, primals_15, reinterpret_tensor(buf23, (32, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf18, (32, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf21, (32, 4, 4), (16, 4, 1), 0), buf39, buf40, reinterpret_tensor(primals_11, (8, 4, 4), (16, 1, 4), 0), primals_6, reinterpret_tensor(buf6, (32, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf2, (32, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf4, (32, 4, 4), (16, 4, 1), 0), buf41, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((8, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_6 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((8, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((8, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((8, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_15 = rand_strided((4, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((16, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, 16, 1), (16, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
