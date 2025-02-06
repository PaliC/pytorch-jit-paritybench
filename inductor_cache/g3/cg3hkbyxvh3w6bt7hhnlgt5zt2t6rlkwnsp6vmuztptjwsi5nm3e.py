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


# kernel path: inductor_cache/6n/c6nqz5claibxllivbz7slkmj6q4aij7fo2634mnxjvkvbabdi7cz.py
# Topologically Sorted Source Nodes: [repeat], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   repeat => repeat
# Graph fragment:
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%primals_1, [4, 1, 1]), kwargs = {})
triton_poi_fused_repeat_0 = async_compile.triton('triton_poi_fused_repeat_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*((x1 % 4))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ge/cgesai5oxgxpkx7jg6e6ppuufdp2o2o2kdi6djicsjglkp2rxl2e.py
# Topologically Sorted Source Nodes: [bmm, bmm_1, bmm_2], Original ATen: [aten.bmm, aten.transpose]
# Source node to ATen node mapping:
#   bmm => bmm
#   bmm_1 => bmm_1
#   bmm_2 => bmm_2
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %primals_2), kwargs = {})
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %primals_3), kwargs = {})
#   %bmm_2 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view, %primals_4), kwargs = {})
#   %permute_11 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view, [0, 2, 1]), kwargs = {})
triton_poi_fused_bmm_transpose_1 = async_compile.triton('triton_poi_fused_bmm_transpose_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_transpose_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_transpose_1(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: inductor_cache/sh/cshimhc6jd4aps5p5ca4g2tp3jpfwjktzmuij3ijs4i6tk5micur.py
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
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'fp64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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


# kernel path: inductor_cache/4z/c4zlfzca2rfnnd73g5fuxyuerqfsf5xr23vtm5bven6jiffkpxus.py
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
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': 'fp64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: inductor_cache/ll/cll2biiez7gesv2a6emfwcuamx56oeprfacedk52nqkdz6gz5pz3.py
# Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   outputs => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%getitem, %getitem_1, %getitem_2, %getitem_3], -1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
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
    tmp19 = tl.load(in_ptr0 + (192 + 4*x1 + ((-12) + x0)), tmp16 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
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
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %expand_3), kwargs = {})
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_3, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_4, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_5, (), ())
    assert_size_stride(primals_6, (4, 16), (16, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [repeat], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_0.run(primals_1, buf0, 256, grid=grid(256), stream=stream0)
        buf1 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        buf5 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        buf16 = empty_strided_cuda((4, 4, 16), (64, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [bmm, bmm_1, bmm_2], Original ATen: [aten.bmm, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_transpose_1.run(buf0, buf1, buf3, buf5, buf16, 256, grid=grid(256), stream=stream0)
        buf2 = reinterpret_tensor(buf0, (4, 16, 4), (64, 4, 1), 0); del buf0  # reuse
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
        buf7 = reinterpret_tensor(buf5, (16, 4, 4), (16, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf4, (16, 4, 4), (16, 1, 4), 0), out=buf7)
        buf8 = empty_strided_cuda((64, 1), (1, 64), torch.float32)
        buf9 = empty_strided_cuda((64, 1), (1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf7, primals_5.item(), buf8, buf9, 64, grid=grid(64), stream=stream0)
        buf10 = reinterpret_tensor(buf7, (64, 4), (4, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf10, primals_5.item(), buf8, buf9, 256, grid=grid(256), stream=stream0)
        buf11 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf6, (16, 4, 4), (16, 4, 1), 0), out=buf11)
        buf12 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf11, buf12, 256, grid=grid(256), stream=stream0)
        del buf11
        buf13 = reinterpret_tensor(buf9, (16, 4), (4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf12, (16, 16), (16, 1), 0), reinterpret_tensor(primals_6, (16, 4), (1, 16), 0), out=buf13)
        buf14 = reinterpret_tensor(buf13, (4, 4, 4), (16, 4, 1), 0); del buf13  # reuse
        # Topologically Sorted Source Nodes: [outputs_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_5.run(buf14, primals_7, primals_1, 64, grid=grid(64), stream=stream0)
        del primals_1
        del primals_7
        buf15 = reinterpret_tensor(buf8, (4, 4, 4), (16, 4, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [sub, add_1, ln_out, mul, ln_out_1], Original ATen: [aten.sub, aten.add, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mul_sub_6.run(buf14, primals_8, primals_9, buf15, 64, grid=grid(64), stream=stream0)
        del primals_9
    return (buf15, primals_5, primals_8, buf10, reinterpret_tensor(buf12, (16, 16), (16, 1), 0), buf14, primals_6, reinterpret_tensor(buf6, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf2, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf4, (16, 4, 4), (16, 4, 1), 0), buf16, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((), (), device='cpu', dtype=torch.float64)
    primals_6 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
