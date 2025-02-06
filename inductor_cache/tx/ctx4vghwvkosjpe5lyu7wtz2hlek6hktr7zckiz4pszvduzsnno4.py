# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/x5/cx54gyrtyz5zrqfp2yli543mnyhxjfzssjghysqku2bnyqvcxfyh.py
# Topologically Sorted Source Nodes: [hx], Original ATen: [aten.zeros]
# Source node to ATen node mapping:
#   hx => full_default
# Graph fragment:
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_0 = async_compile.triton('triton_poi_fused_zeros_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/es/cesjb7wb5kf6ucofqzniwk3hhjvdmv2s75fytyfw4r3qxvtxsacy.py
# Topologically Sorted Source Nodes: [res, res_1], Original ATen: [aten.add, aten.tanh]
# Source node to ATen node mapping:
#   res => add
#   res_1 => tanh
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %unsqueeze_1), kwargs = {})
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add,), kwargs = {})
triton_poi_fused_add_tanh_1 = async_compile.triton('triton_poi_fused_add_tanh_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_tanh_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_tanh_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jg/cjgo2pyx2uddnbdl225cxepn4dke2jxdcvs266ieyjtrdgtf63vs.py
# Topologically Sorted Source Nodes: [alpha], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   alpha => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_3, [1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_2 = async_compile.triton('triton_poi_fused__softmax_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/in/cinemwlxrwtrmuaev4jit5azxxaplv3xmwsptbua5h55rxhaggth.py
# Topologically Sorted Source Nodes: [alpha], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   alpha => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_3 = async_compile.triton('triton_poi_fused__softmax_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kn/ckn4qxsirkvuncp7ry7cicnyufuxa5q4d3nizbu2ufljvl76nhfw.py
# Topologically Sorted Source Nodes: [concat_context], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concat_context => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%squeeze, %convert_element_type_1], 1), kwargs = {})
triton_poi_fused_cat_4 = async_compile.triton('triton_poi_fused_cat_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
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
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = (-4) + x0
    tmp11 = tmp9 == tmp10
    tmp12 = tmp11.to(tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vj/cvj2mqaaqnrpmmok7bdkrb3wxmr4c7gaqkxy6xali4mkxdkvuwta.py
# Topologically Sorted Source Nodes: [next_input], Original ATen: [aten.argmax]
# Source node to ATen node mapping:
#   next_input => argmax
# Graph fragment:
#   %argmax : [num_users=1] = call_function[target=torch.ops.aten.argmax.default](args = (%addmm_1, 1), kwargs = {})
triton_poi_fused_argmax_5 = async_compile.triton('triton_poi_fused_argmax_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_argmax_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_argmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/xd/cxdfdgnrdxlsdahnnthuxefyxfjo3g4fbgr5o5q5dfpuuhixohu4.py
# Topologically Sorted Source Nodes: [concat_context_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   concat_context_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%squeeze_1, %convert_element_type_2], 1), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
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
    tmp9 = tl.load(in_ptr1 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = (-4) + x0
    tmp11 = tmp9 == tmp10
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp6, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dd/cdd7a7n4phs37rsdvr74yamotilmzgzti3ely26jx5nix7ozys2e.py
# Topologically Sorted Source Nodes: [res_48, res_49], Original ATen: [aten.add, aten.tanh]
# Source node to ATen node mapping:
#   res_48 => add_24
#   res_49 => tanh_24
# Graph fragment:
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %unsqueeze_49), kwargs = {})
#   %tanh_24 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%add_24,), kwargs = {})
triton_poi_fused_add_tanh_7 = async_compile.triton('triton_poi_fused_add_tanh_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_tanh_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_tanh_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = libdevice.tanh(tmp4)
    tl.store(in_out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6f/c6fdzjzmt7gqxlovbw5sthdvowhnusmqumep7vdd2i2y4fzz5pie.py
# Topologically Sorted Source Nodes: [probs, setitem, setitem_1, setitem_2, setitem_3, setitem_4, setitem_5, setitem_6, setitem_7, setitem_8, setitem_9, setitem_10, setitem_11, setitem_12, setitem_13, setitem_14, setitem_15, setitem_16, setitem_17, setitem_18, setitem_19, setitem_20, setitem_21, setitem_22, setitem_23, setitem_24], Original ATen: [aten.zeros, aten.copy]
# Source node to ATen node mapping:
#   probs => full_default_1
#   setitem => copy
#   setitem_1 => copy_1
#   setitem_10 => copy_10
#   setitem_11 => copy_11
#   setitem_12 => copy_12
#   setitem_13 => copy_13
#   setitem_14 => copy_14
#   setitem_15 => copy_15
#   setitem_16 => copy_16
#   setitem_17 => copy_17
#   setitem_18 => copy_18
#   setitem_19 => copy_19
#   setitem_2 => copy_2
#   setitem_20 => copy_20
#   setitem_21 => copy_21
#   setitem_22 => copy_22
#   setitem_23 => copy_23
#   setitem_24 => copy_24
#   setitem_3 => copy_3
#   setitem_4 => copy_4
#   setitem_5 => copy_5
#   setitem_6 => copy_6
#   setitem_7 => copy_7
#   setitem_8 => copy_8
#   setitem_9 => copy_9
# Graph fragment:
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([4, 25, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %copy : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select, %addmm_1), kwargs = {})
#   %select_scatter_default : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%full_default_1, %copy, 1, 0), kwargs = {})
#   %copy_1 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_5, %addmm_3), kwargs = {})
#   %select_scatter_default_1 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default, %copy_1, 1, 1), kwargs = {})
#   %copy_2 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_10, %addmm_5), kwargs = {})
#   %select_scatter_default_2 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_1, %copy_2, 1, 2), kwargs = {})
#   %copy_3 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_15, %addmm_7), kwargs = {})
#   %select_scatter_default_3 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_2, %copy_3, 1, 3), kwargs = {})
#   %copy_4 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_20, %addmm_9), kwargs = {})
#   %select_scatter_default_4 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_3, %copy_4, 1, 4), kwargs = {})
#   %copy_5 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_25, %addmm_11), kwargs = {})
#   %select_scatter_default_5 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_4, %copy_5, 1, 5), kwargs = {})
#   %copy_6 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_30, %addmm_13), kwargs = {})
#   %select_scatter_default_6 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_5, %copy_6, 1, 6), kwargs = {})
#   %copy_7 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_35, %addmm_15), kwargs = {})
#   %select_scatter_default_7 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_6, %copy_7, 1, 7), kwargs = {})
#   %copy_8 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_40, %addmm_17), kwargs = {})
#   %select_scatter_default_8 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_7, %copy_8, 1, 8), kwargs = {})
#   %copy_9 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_45, %addmm_19), kwargs = {})
#   %select_scatter_default_9 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_8, %copy_9, 1, 9), kwargs = {})
#   %copy_10 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_50, %addmm_21), kwargs = {})
#   %select_scatter_default_10 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_9, %copy_10, 1, 10), kwargs = {})
#   %copy_11 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_55, %addmm_23), kwargs = {})
#   %select_scatter_default_11 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_10, %copy_11, 1, 11), kwargs = {})
#   %copy_12 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_60, %addmm_25), kwargs = {})
#   %select_scatter_default_12 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_11, %copy_12, 1, 12), kwargs = {})
#   %copy_13 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_65, %addmm_27), kwargs = {})
#   %select_scatter_default_13 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_12, %copy_13, 1, 13), kwargs = {})
#   %copy_14 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_70, %addmm_29), kwargs = {})
#   %select_scatter_default_14 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_13, %copy_14, 1, 14), kwargs = {})
#   %copy_15 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_75, %addmm_31), kwargs = {})
#   %select_scatter_default_15 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_14, %copy_15, 1, 15), kwargs = {})
#   %copy_16 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_80, %addmm_33), kwargs = {})
#   %select_scatter_default_16 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_15, %copy_16, 1, 16), kwargs = {})
#   %copy_17 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_85, %addmm_35), kwargs = {})
#   %select_scatter_default_17 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_16, %copy_17, 1, 17), kwargs = {})
#   %copy_18 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_90, %addmm_37), kwargs = {})
#   %select_scatter_default_18 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_17, %copy_18, 1, 18), kwargs = {})
#   %copy_19 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_95, %addmm_39), kwargs = {})
#   %select_scatter_default_19 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_18, %copy_19, 1, 19), kwargs = {})
#   %copy_20 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_100, %addmm_41), kwargs = {})
#   %select_scatter_default_20 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_19, %copy_20, 1, 20), kwargs = {})
#   %copy_21 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_105, %addmm_43), kwargs = {})
#   %select_scatter_default_21 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_20, %copy_21, 1, 21), kwargs = {})
#   %copy_22 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_110, %addmm_45), kwargs = {})
#   %select_scatter_default_22 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_21, %copy_22, 1, 22), kwargs = {})
#   %copy_23 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_115, %addmm_47), kwargs = {})
#   %select_scatter_default_23 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_22, %copy_23, 1, 23), kwargs = {})
#   %copy_24 : [num_users=1] = call_function[target=torch.ops.aten.copy.default](args = (%select_120, %addmm_49), kwargs = {})
#   %select_scatter_default_24 : [num_users=2] = call_function[target=torch.ops.aten.select_scatter.default](args = (%select_scatter_default_23, %copy_24, 1, 24), kwargs = {})
triton_poi_fused_copy_zeros_8 = async_compile.triton('triton_poi_fused_copy_zeros_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'in_ptr20': '*fp32', 'in_ptr21': '*fp32', 'in_ptr22': '*fp32', 'in_ptr23': '*fp32', 'in_ptr24': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_copy_zeros_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 25, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_copy_zeros_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 25)
    x0 = (xindex % 4)
    x2 = xindex // 100
    x3 = xindex
    tmp3 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr5 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr6 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr7 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr9 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr10 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr11 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr12 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr13 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr14 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr15 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr16 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr17 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr18 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr19 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp81 = tl.load(in_ptr20 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp88 = tl.load(in_ptr21 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr22 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp94 = tl.load(in_ptr23 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr24 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 4, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tl.full([1], 3, tl.int32)
    tmp5 = tmp0 == tmp4
    tmp7 = tl.full([1], 2, tl.int32)
    tmp8 = tmp0 == tmp7
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp0 == tmp10
    tmp13 = tl.full([1], 0, tl.int32)
    tmp14 = tmp0 == tmp13
    tmp16 = 0.0
    tmp17 = tl.where(tmp14, tmp15, tmp16)
    tmp18 = tl.where(tmp11, tmp12, tmp17)
    tmp19 = tl.where(tmp8, tmp9, tmp18)
    tmp20 = tl.where(tmp5, tmp6, tmp19)
    tmp21 = tl.where(tmp2, tmp3, tmp20)
    tmp22 = tl.full([1], 8, tl.int32)
    tmp23 = tmp0 == tmp22
    tmp25 = tl.full([1], 7, tl.int32)
    tmp26 = tmp0 == tmp25
    tmp28 = tl.full([1], 6, tl.int32)
    tmp29 = tmp0 == tmp28
    tmp31 = tl.full([1], 5, tl.int32)
    tmp32 = tmp0 == tmp31
    tmp34 = tl.where(tmp32, tmp33, tmp21)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp26, tmp27, tmp35)
    tmp37 = tl.where(tmp23, tmp24, tmp36)
    tmp38 = tl.full([1], 12, tl.int32)
    tmp39 = tmp0 == tmp38
    tmp41 = tl.full([1], 11, tl.int32)
    tmp42 = tmp0 == tmp41
    tmp44 = tl.full([1], 10, tl.int32)
    tmp45 = tmp0 == tmp44
    tmp47 = tl.full([1], 9, tl.int32)
    tmp48 = tmp0 == tmp47
    tmp50 = tl.where(tmp48, tmp49, tmp37)
    tmp51 = tl.where(tmp45, tmp46, tmp50)
    tmp52 = tl.where(tmp42, tmp43, tmp51)
    tmp53 = tl.where(tmp39, tmp40, tmp52)
    tmp54 = tl.full([1], 16, tl.int32)
    tmp55 = tmp0 == tmp54
    tmp57 = tl.full([1], 15, tl.int32)
    tmp58 = tmp0 == tmp57
    tmp60 = tl.full([1], 14, tl.int32)
    tmp61 = tmp0 == tmp60
    tmp63 = tl.full([1], 13, tl.int32)
    tmp64 = tmp0 == tmp63
    tmp66 = tl.where(tmp64, tmp65, tmp53)
    tmp67 = tl.where(tmp61, tmp62, tmp66)
    tmp68 = tl.where(tmp58, tmp59, tmp67)
    tmp69 = tl.where(tmp55, tmp56, tmp68)
    tmp70 = tl.full([1], 20, tl.int32)
    tmp71 = tmp0 == tmp70
    tmp73 = tl.full([1], 19, tl.int32)
    tmp74 = tmp0 == tmp73
    tmp76 = tl.full([1], 18, tl.int32)
    tmp77 = tmp0 == tmp76
    tmp79 = tl.full([1], 17, tl.int32)
    tmp80 = tmp0 == tmp79
    tmp82 = tl.where(tmp80, tmp81, tmp69)
    tmp83 = tl.where(tmp77, tmp78, tmp82)
    tmp84 = tl.where(tmp74, tmp75, tmp83)
    tmp85 = tl.where(tmp71, tmp72, tmp84)
    tmp86 = tl.full([1], 24, tl.int32)
    tmp87 = tmp0 == tmp86
    tmp89 = tl.full([1], 23, tl.int32)
    tmp90 = tmp0 == tmp89
    tmp92 = tl.full([1], 22, tl.int32)
    tmp93 = tmp0 == tmp92
    tmp95 = tl.full([1], 21, tl.int32)
    tmp96 = tmp0 == tmp95
    tmp98 = tl.where(tmp96, tmp97, tmp85)
    tmp99 = tl.where(tmp93, tmp94, tmp98)
    tmp100 = tl.where(tmp90, tmp91, tmp99)
    tmp101 = tl.where(tmp87, tmp88, tmp100)
    tl.store(in_out_ptr0 + (x3), tmp101, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/su/csu4woziskojvir735cp7ss3cdr7vf4nmpsb47mjgvk4uysnyd4o.py
# Topologically Sorted Source Nodes: [probs_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   probs_1 => amax_25, exp_25, sub_25
# Graph fragment:
#   %amax_25 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%select_scatter_default_24, [2], True), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%select_scatter_default_24, %amax_25), kwargs = {})
#   %exp_25 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_25,), kwargs = {})
triton_poi_fused__softmax_9 = async_compile.triton('triton_poi_fused__softmax_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
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
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ny/cny6q5u6dnkven2drqpnh5wqvhvgnevua4xpuadwbi2y2u25gskv.py
# Topologically Sorted Source Nodes: [probs_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   probs_1 => div_25, sum_26
# Graph fragment:
#   %sum_26 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_25, [2], True), kwargs = {})
#   %div_25 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_25, %sum_26), kwargs = {})
triton_poi_fused__softmax_10 = async_compile.triton('triton_poi_fused__softmax_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
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
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (1, 4), (4, 1))
    assert_size_stride(primals_6, (12, 8), (8, 1))
    assert_size_stride(primals_7, (12, 4), (4, 1))
    assert_size_stride(primals_8, (12, ), (1, ))
    assert_size_stride(primals_9, (12, ), (1, ))
    assert_size_stride(primals_10, (4, 4), (4, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [hx], Original ATen: [aten.zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_0.run(buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [batch_H_proj], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (16, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf1)
        del primals_2
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf2)
        buf3 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res, res_1], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf2, primals_4, buf3, 64, grid=grid(64), stream=stream0)
        buf4 = reinterpret_tensor(buf2, (16, 1), (1, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [e], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf4)
        buf5 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf4, buf5, 16, grid=grid(16), stream=stream0)
        buf6 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf5, buf6, 16, grid=grid(16), stream=stream0)
        buf7 = reinterpret_tensor(buf5, (4, 1, 4), (4, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf7)
        buf8 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_4.run(buf7, buf8, 32, grid=grid(32), stream=stream0)
        buf9 = empty_strided_cuda((4, 12), (12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf9)
        buf10 = empty_strided_cuda((4, 12), (12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf10)
        # Topologically Sorted Source Nodes: [ret], Original ATen: [aten._thnn_fused_gru_cell]
        buf11 = torch.ops.aten._thnn_fused_gru_cell.default(buf9, buf10, buf0, primals_8, primals_9)
        buf12 = buf11[0]
        buf13 = buf11[1]
        del buf11
        buf14 = reinterpret_tensor(buf7, (4, 4), (4, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [probs_step], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf12, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf14)
        buf15 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [next_input], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf14, buf15, 4, grid=grid(4), stream=stream0)
        buf16 = reinterpret_tensor(buf6, (4, 4), (4, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf16)
        buf17 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_2, res_3], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf16, primals_4, buf17, 64, grid=grid(64), stream=stream0)
        buf18 = reinterpret_tensor(buf16, (16, 1), (1, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [e_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf17, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf18)
        buf19 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf18, buf19, 16, grid=grid(16), stream=stream0)
        buf20 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_2], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf19, buf20, 16, grid=grid(16), stream=stream0)
        buf21 = reinterpret_tensor(buf19, (4, 1, 4), (4, 4, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf21)
        buf22 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf21, buf15, buf22, 32, grid=grid(32), stream=stream0)
        buf23 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf22, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf23)
        buf24 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf12, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf24)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten._thnn_fused_gru_cell]
        buf25 = torch.ops.aten._thnn_fused_gru_cell.default(buf23, buf24, buf12, primals_8, primals_9)
        buf26 = buf25[0]
        buf27 = buf25[1]
        del buf25
        buf28 = reinterpret_tensor(buf21, (4, 4), (4, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [probs_step_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf26, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf28)
        buf29 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [next_input_1], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf28, buf29, 4, grid=grid(4), stream=stream0)
        buf30 = reinterpret_tensor(buf20, (4, 4), (4, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.mm(buf26, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf30)
        buf31 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_4, res_5], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf30, primals_4, buf31, 64, grid=grid(64), stream=stream0)
        buf32 = reinterpret_tensor(buf30, (16, 1), (1, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [e_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf31, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf32)
        buf33 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf32, buf33, 16, grid=grid(16), stream=stream0)
        buf34 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf33, buf34, 16, grid=grid(16), stream=stream0)
        buf35 = reinterpret_tensor(buf33, (4, 1, 4), (4, 4, 1), 0); del buf33  # reuse
        # Topologically Sorted Source Nodes: [bmm_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf34, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf35)
        buf36 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf35, buf29, buf36, 32, grid=grid(32), stream=stream0)
        buf37 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf36, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf37)
        buf38 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf26, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf38)
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten._thnn_fused_gru_cell]
        buf39 = torch.ops.aten._thnn_fused_gru_cell.default(buf37, buf38, buf26, primals_8, primals_9)
        buf40 = buf39[0]
        buf41 = buf39[1]
        del buf39
        buf42 = reinterpret_tensor(buf35, (4, 4), (4, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [probs_step_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf40, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf42)
        buf43 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [next_input_2], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf42, buf43, 4, grid=grid(4), stream=stream0)
        buf44 = reinterpret_tensor(buf34, (4, 4), (4, 1), 0); del buf34  # reuse
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf44)
        buf45 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_6, res_7], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf44, primals_4, buf45, 64, grid=grid(64), stream=stream0)
        buf46 = reinterpret_tensor(buf44, (16, 1), (1, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [e_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf46)
        buf47 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_6], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf46, buf47, 16, grid=grid(16), stream=stream0)
        buf48 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_6], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf47, buf48, 16, grid=grid(16), stream=stream0)
        buf49 = reinterpret_tensor(buf47, (4, 1, 4), (4, 4, 1), 0); del buf47  # reuse
        # Topologically Sorted Source Nodes: [bmm_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf48, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf49)
        buf50 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_3], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf49, buf43, buf50, 32, grid=grid(32), stream=stream0)
        buf51 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf50, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf51)
        buf52 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf52)
        # Topologically Sorted Source Nodes: [ret_3], Original ATen: [aten._thnn_fused_gru_cell]
        buf53 = torch.ops.aten._thnn_fused_gru_cell.default(buf51, buf52, buf40, primals_8, primals_9)
        buf54 = buf53[0]
        buf55 = buf53[1]
        del buf53
        buf56 = reinterpret_tensor(buf49, (4, 4), (4, 1), 0); del buf49  # reuse
        # Topologically Sorted Source Nodes: [probs_step_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf54, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf56)
        buf57 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [next_input_3], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf56, buf57, 4, grid=grid(4), stream=stream0)
        buf58 = reinterpret_tensor(buf48, (4, 4), (4, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.addmm]
        extern_kernels.mm(buf54, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf58)
        buf59 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_8, res_9], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf58, primals_4, buf59, 64, grid=grid(64), stream=stream0)
        buf60 = reinterpret_tensor(buf58, (16, 1), (1, 1), 0); del buf58  # reuse
        # Topologically Sorted Source Nodes: [e_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf60)
        buf61 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_8], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf60, buf61, 16, grid=grid(16), stream=stream0)
        buf62 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_8], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf61, buf62, 16, grid=grid(16), stream=stream0)
        buf63 = reinterpret_tensor(buf61, (4, 1, 4), (4, 4, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [bmm_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf62, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf63)
        buf64 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf63, buf57, buf64, 32, grid=grid(32), stream=stream0)
        buf65 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf64, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf65)
        buf66 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf54, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf66)
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten._thnn_fused_gru_cell]
        buf67 = torch.ops.aten._thnn_fused_gru_cell.default(buf65, buf66, buf54, primals_8, primals_9)
        buf68 = buf67[0]
        buf69 = buf67[1]
        del buf67
        buf70 = reinterpret_tensor(buf63, (4, 4), (4, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [probs_step_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf68, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf70)
        buf72 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [next_input_4], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf70, buf72, 4, grid=grid(4), stream=stream0)
        buf73 = reinterpret_tensor(buf62, (4, 4), (4, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
        extern_kernels.mm(buf68, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf73)
        buf74 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_10, res_11], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf73, primals_4, buf74, 64, grid=grid(64), stream=stream0)
        buf75 = reinterpret_tensor(buf73, (16, 1), (1, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [e_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf74, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf75)
        buf76 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_10], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf75, buf76, 16, grid=grid(16), stream=stream0)
        buf77 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_10], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf76, buf77, 16, grid=grid(16), stream=stream0)
        buf78 = reinterpret_tensor(buf76, (4, 1, 4), (4, 4, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [bmm_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf78)
        buf79 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_5], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf78, buf72, buf79, 32, grid=grid(32), stream=stream0)
        buf80 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf79, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf80)
        buf81 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf68, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf81)
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten._thnn_fused_gru_cell]
        buf82 = torch.ops.aten._thnn_fused_gru_cell.default(buf80, buf81, buf68, primals_8, primals_9)
        buf83 = buf82[0]
        buf85 = reinterpret_tensor(buf78, (4, 4), (4, 1), 0); del buf78  # reuse
        # Topologically Sorted Source Nodes: [probs_step_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf83, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf85)
        buf86 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [next_input_5], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf85, buf86, 4, grid=grid(4), stream=stream0)
        buf87 = reinterpret_tensor(buf77, (4, 4), (4, 1), 0); del buf77  # reuse
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.mm(buf83, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf87)
        buf88 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_12, res_13], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf87, primals_4, buf88, 64, grid=grid(64), stream=stream0)
        buf89 = reinterpret_tensor(buf87, (16, 1), (1, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [e_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf88, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf89)
        buf90 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_12], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf89, buf90, 16, grid=grid(16), stream=stream0)
        buf91 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_12], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf90, buf91, 16, grid=grid(16), stream=stream0)
        buf92 = reinterpret_tensor(buf90, (4, 1, 4), (4, 4, 1), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [bmm_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf91, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf92)
        buf93 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_6], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf92, buf86, buf93, 32, grid=grid(32), stream=stream0)
        buf94 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf93, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf94)
        buf95 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf83, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf95)
        # Topologically Sorted Source Nodes: [ret_6], Original ATen: [aten._thnn_fused_gru_cell]
        buf96 = torch.ops.aten._thnn_fused_gru_cell.default(buf94, buf95, buf83, primals_8, primals_9)
        buf97 = buf96[0]
        buf99 = reinterpret_tensor(buf92, (4, 4), (4, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [probs_step_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf97, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf99)
        buf100 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [next_input_6], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf99, buf100, 4, grid=grid(4), stream=stream0)
        buf101 = reinterpret_tensor(buf91, (4, 4), (4, 1), 0); del buf91  # reuse
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.addmm]
        extern_kernels.mm(buf97, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf101)
        buf102 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_14, res_15], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf101, primals_4, buf102, 64, grid=grid(64), stream=stream0)
        buf103 = reinterpret_tensor(buf101, (16, 1), (1, 1), 0); del buf101  # reuse
        # Topologically Sorted Source Nodes: [e_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf103)
        buf104 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_14], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf103, buf104, 16, grid=grid(16), stream=stream0)
        buf105 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_14], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf104, buf105, 16, grid=grid(16), stream=stream0)
        buf106 = reinterpret_tensor(buf104, (4, 1, 4), (4, 4, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [bmm_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf106)
        buf107 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf106, buf100, buf107, 32, grid=grid(32), stream=stream0)
        buf108 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf107, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf108)
        buf109 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten.mm]
        extern_kernels.mm(buf97, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf109)
        # Topologically Sorted Source Nodes: [ret_7], Original ATen: [aten._thnn_fused_gru_cell]
        buf110 = torch.ops.aten._thnn_fused_gru_cell.default(buf108, buf109, buf97, primals_8, primals_9)
        buf111 = buf110[0]
        buf113 = reinterpret_tensor(buf106, (4, 4), (4, 1), 0); del buf106  # reuse
        # Topologically Sorted Source Nodes: [probs_step_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf111, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf113)
        buf114 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [next_input_7], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf113, buf114, 4, grid=grid(4), stream=stream0)
        buf115 = reinterpret_tensor(buf105, (4, 4), (4, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.addmm]
        extern_kernels.mm(buf111, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf115)
        buf116 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_16, res_17], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf115, primals_4, buf116, 64, grid=grid(64), stream=stream0)
        buf117 = reinterpret_tensor(buf115, (16, 1), (1, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [e_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf117)
        buf118 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_16], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf117, buf118, 16, grid=grid(16), stream=stream0)
        buf119 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_16], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf118, buf119, 16, grid=grid(16), stream=stream0)
        buf120 = reinterpret_tensor(buf118, (4, 1, 4), (4, 4, 1), 0); del buf118  # reuse
        # Topologically Sorted Source Nodes: [bmm_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf119, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf120)
        buf121 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_8], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf120, buf114, buf121, 32, grid=grid(32), stream=stream0)
        buf122 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf121, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf122)
        buf123 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf111, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf123)
        # Topologically Sorted Source Nodes: [ret_8], Original ATen: [aten._thnn_fused_gru_cell]
        buf124 = torch.ops.aten._thnn_fused_gru_cell.default(buf122, buf123, buf111, primals_8, primals_9)
        buf125 = buf124[0]
        buf127 = reinterpret_tensor(buf120, (4, 4), (4, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [probs_step_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf125, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf127)
        buf129 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [next_input_8], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf127, buf129, 4, grid=grid(4), stream=stream0)
        buf130 = reinterpret_tensor(buf119, (4, 4), (4, 1), 0); del buf119  # reuse
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.mm(buf125, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf130)
        buf131 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_18, res_19], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf130, primals_4, buf131, 64, grid=grid(64), stream=stream0)
        buf132 = reinterpret_tensor(buf130, (16, 1), (1, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [e_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf131, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf132)
        buf133 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_18], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf132, buf133, 16, grid=grid(16), stream=stream0)
        buf134 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_18], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf133, buf134, 16, grid=grid(16), stream=stream0)
        buf135 = reinterpret_tensor(buf133, (4, 1, 4), (4, 4, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [bmm_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf134, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf135)
        buf136 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_9], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf135, buf129, buf136, 32, grid=grid(32), stream=stream0)
        buf137 = buf123; del buf123  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf136, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf137)
        buf138 = buf122; del buf122  # reuse
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten.mm]
        extern_kernels.mm(buf125, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf138)
        # Topologically Sorted Source Nodes: [ret_9], Original ATen: [aten._thnn_fused_gru_cell]
        buf139 = torch.ops.aten._thnn_fused_gru_cell.default(buf137, buf138, buf125, primals_8, primals_9)
        buf140 = buf139[0]
        buf142 = reinterpret_tensor(buf135, (4, 4), (4, 1), 0); del buf135  # reuse
        # Topologically Sorted Source Nodes: [probs_step_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf140, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf142)
        buf143 = buf129; del buf129  # reuse
        # Topologically Sorted Source Nodes: [next_input_9], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf142, buf143, 4, grid=grid(4), stream=stream0)
        buf144 = reinterpret_tensor(buf134, (4, 4), (4, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.addmm]
        extern_kernels.mm(buf140, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf144)
        buf145 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_20, res_21], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf144, primals_4, buf145, 64, grid=grid(64), stream=stream0)
        buf146 = reinterpret_tensor(buf144, (16, 1), (1, 1), 0); del buf144  # reuse
        # Topologically Sorted Source Nodes: [e_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf145, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf146)
        buf147 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_20], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf146, buf147, 16, grid=grid(16), stream=stream0)
        buf148 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_20], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf147, buf148, 16, grid=grid(16), stream=stream0)
        buf149 = reinterpret_tensor(buf147, (4, 1, 4), (4, 4, 1), 0); del buf147  # reuse
        # Topologically Sorted Source Nodes: [bmm_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf148, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf149)
        buf150 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_10], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf149, buf143, buf150, 32, grid=grid(32), stream=stream0)
        buf151 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf150, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf151)
        buf152 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf140, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf152)
        # Topologically Sorted Source Nodes: [ret_10], Original ATen: [aten._thnn_fused_gru_cell]
        buf153 = torch.ops.aten._thnn_fused_gru_cell.default(buf151, buf152, buf140, primals_8, primals_9)
        buf154 = buf153[0]
        buf156 = reinterpret_tensor(buf149, (4, 4), (4, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [probs_step_10], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf154, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf156)
        buf157 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [next_input_10], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf156, buf157, 4, grid=grid(4), stream=stream0)
        buf158 = reinterpret_tensor(buf148, (4, 4), (4, 1), 0); del buf148  # reuse
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.addmm]
        extern_kernels.mm(buf154, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf158)
        buf159 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_22, res_23], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf158, primals_4, buf159, 64, grid=grid(64), stream=stream0)
        buf160 = reinterpret_tensor(buf158, (16, 1), (1, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [e_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf160)
        buf161 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_22], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf160, buf161, 16, grid=grid(16), stream=stream0)
        buf162 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_22], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf161, buf162, 16, grid=grid(16), stream=stream0)
        buf163 = reinterpret_tensor(buf161, (4, 1, 4), (4, 4, 1), 0); del buf161  # reuse
        # Topologically Sorted Source Nodes: [bmm_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf162, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf163)
        buf164 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_11], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf163, buf157, buf164, 32, grid=grid(32), stream=stream0)
        buf165 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf164, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf165)
        buf166 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten.mm]
        extern_kernels.mm(buf154, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf166)
        # Topologically Sorted Source Nodes: [ret_11], Original ATen: [aten._thnn_fused_gru_cell]
        buf167 = torch.ops.aten._thnn_fused_gru_cell.default(buf165, buf166, buf154, primals_8, primals_9)
        buf168 = buf167[0]
        buf170 = reinterpret_tensor(buf163, (4, 4), (4, 1), 0); del buf163  # reuse
        # Topologically Sorted Source Nodes: [probs_step_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf168, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf170)
        buf171 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [next_input_11], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf170, buf171, 4, grid=grid(4), stream=stream0)
        buf172 = reinterpret_tensor(buf162, (4, 4), (4, 1), 0); del buf162  # reuse
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.mm(buf168, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf172)
        buf173 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_24, res_25], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf172, primals_4, buf173, 64, grid=grid(64), stream=stream0)
        buf174 = reinterpret_tensor(buf172, (16, 1), (1, 1), 0); del buf172  # reuse
        # Topologically Sorted Source Nodes: [e_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf174)
        buf175 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_24], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf174, buf175, 16, grid=grid(16), stream=stream0)
        buf176 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_24], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf175, buf176, 16, grid=grid(16), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (4, 1, 4), (4, 4, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [bmm_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf177)
        buf178 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_12], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf177, buf171, buf178, 32, grid=grid(32), stream=stream0)
        buf179 = buf166; del buf166  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf178, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf179)
        buf180 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf168, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf180)
        # Topologically Sorted Source Nodes: [ret_12], Original ATen: [aten._thnn_fused_gru_cell]
        buf181 = torch.ops.aten._thnn_fused_gru_cell.default(buf179, buf180, buf168, primals_8, primals_9)
        buf182 = buf181[0]
        buf184 = reinterpret_tensor(buf177, (4, 4), (4, 1), 0); del buf177  # reuse
        # Topologically Sorted Source Nodes: [probs_step_12], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf182, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf184)
        buf186 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [next_input_12], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf184, buf186, 4, grid=grid(4), stream=stream0)
        buf187 = reinterpret_tensor(buf176, (4, 4), (4, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.addmm]
        extern_kernels.mm(buf182, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf187)
        buf188 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_26, res_27], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf187, primals_4, buf188, 64, grid=grid(64), stream=stream0)
        buf189 = reinterpret_tensor(buf187, (16, 1), (1, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [e_13], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf189)
        buf190 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_26], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf189, buf190, 16, grid=grid(16), stream=stream0)
        buf191 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_26], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf190, buf191, 16, grid=grid(16), stream=stream0)
        buf192 = reinterpret_tensor(buf190, (4, 1, 4), (4, 4, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [bmm_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf192)
        buf193 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_13], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf192, buf186, buf193, 32, grid=grid(32), stream=stream0)
        buf194 = buf180; del buf180  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf193, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf194)
        buf195 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten.mm]
        extern_kernels.mm(buf182, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf195)
        # Topologically Sorted Source Nodes: [ret_13], Original ATen: [aten._thnn_fused_gru_cell]
        buf196 = torch.ops.aten._thnn_fused_gru_cell.default(buf194, buf195, buf182, primals_8, primals_9)
        buf197 = buf196[0]
        buf199 = reinterpret_tensor(buf192, (4, 4), (4, 1), 0); del buf192  # reuse
        # Topologically Sorted Source Nodes: [probs_step_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf197, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf199)
        buf200 = buf186; del buf186  # reuse
        # Topologically Sorted Source Nodes: [next_input_13], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf199, buf200, 4, grid=grid(4), stream=stream0)
        buf201 = reinterpret_tensor(buf191, (4, 4), (4, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.mm(buf197, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf201)
        buf202 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_28, res_29], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf201, primals_4, buf202, 64, grid=grid(64), stream=stream0)
        buf203 = reinterpret_tensor(buf201, (16, 1), (1, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [e_14], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf203)
        buf204 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_28], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf203, buf204, 16, grid=grid(16), stream=stream0)
        buf205 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_28], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf204, buf205, 16, grid=grid(16), stream=stream0)
        buf206 = reinterpret_tensor(buf204, (4, 1, 4), (4, 4, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [bmm_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf206)
        buf207 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_14], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf206, buf200, buf207, 32, grid=grid(32), stream=stream0)
        buf208 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf207, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf208)
        buf209 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf197, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf209)
        # Topologically Sorted Source Nodes: [ret_14], Original ATen: [aten._thnn_fused_gru_cell]
        buf210 = torch.ops.aten._thnn_fused_gru_cell.default(buf208, buf209, buf197, primals_8, primals_9)
        buf211 = buf210[0]
        buf213 = reinterpret_tensor(buf206, (4, 4), (4, 1), 0); del buf206  # reuse
        # Topologically Sorted Source Nodes: [probs_step_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf211, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf213)
        buf214 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [next_input_14], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf213, buf214, 4, grid=grid(4), stream=stream0)
        buf215 = reinterpret_tensor(buf205, (4, 4), (4, 1), 0); del buf205  # reuse
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.mm(buf211, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf215)
        buf216 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_30, res_31], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf215, primals_4, buf216, 64, grid=grid(64), stream=stream0)
        buf217 = reinterpret_tensor(buf215, (16, 1), (1, 1), 0); del buf215  # reuse
        # Topologically Sorted Source Nodes: [e_15], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf216, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf217)
        buf218 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_30], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf217, buf218, 16, grid=grid(16), stream=stream0)
        buf219 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_30], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf218, buf219, 16, grid=grid(16), stream=stream0)
        buf220 = reinterpret_tensor(buf218, (4, 1, 4), (4, 4, 1), 0); del buf218  # reuse
        # Topologically Sorted Source Nodes: [bmm_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf219, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf220)
        buf221 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_15], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf220, buf214, buf221, 32, grid=grid(32), stream=stream0)
        buf222 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf221, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf222)
        buf223 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf211, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf223)
        # Topologically Sorted Source Nodes: [ret_15], Original ATen: [aten._thnn_fused_gru_cell]
        buf224 = torch.ops.aten._thnn_fused_gru_cell.default(buf222, buf223, buf211, primals_8, primals_9)
        buf225 = buf224[0]
        buf227 = reinterpret_tensor(buf220, (4, 4), (4, 1), 0); del buf220  # reuse
        # Topologically Sorted Source Nodes: [probs_step_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf225, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf227)
        buf228 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [next_input_15], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf227, buf228, 4, grid=grid(4), stream=stream0)
        buf229 = reinterpret_tensor(buf219, (4, 4), (4, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [linear_65], Original ATen: [aten.addmm]
        extern_kernels.mm(buf225, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf229)
        buf230 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_32, res_33], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf229, primals_4, buf230, 64, grid=grid(64), stream=stream0)
        buf231 = reinterpret_tensor(buf229, (16, 1), (1, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [e_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf230, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf231)
        buf232 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_32], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf231, buf232, 16, grid=grid(16), stream=stream0)
        buf233 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_32], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf232, buf233, 16, grid=grid(16), stream=stream0)
        buf234 = reinterpret_tensor(buf232, (4, 1, 4), (4, 4, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [bmm_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf233, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf234)
        buf235 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_16], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf234, buf228, buf235, 32, grid=grid(32), stream=stream0)
        buf236 = buf223; del buf223  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf235, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf236)
        buf237 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf225, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf237)
        # Topologically Sorted Source Nodes: [ret_16], Original ATen: [aten._thnn_fused_gru_cell]
        buf238 = torch.ops.aten._thnn_fused_gru_cell.default(buf236, buf237, buf225, primals_8, primals_9)
        buf239 = buf238[0]
        buf241 = reinterpret_tensor(buf234, (4, 4), (4, 1), 0); del buf234  # reuse
        # Topologically Sorted Source Nodes: [probs_step_16], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf239, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf241)
        buf243 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [next_input_16], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf241, buf243, 4, grid=grid(4), stream=stream0)
        buf244 = reinterpret_tensor(buf233, (4, 4), (4, 1), 0); del buf233  # reuse
        # Topologically Sorted Source Nodes: [linear_69], Original ATen: [aten.addmm]
        extern_kernels.mm(buf239, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf244)
        buf245 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_34, res_35], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf244, primals_4, buf245, 64, grid=grid(64), stream=stream0)
        buf246 = reinterpret_tensor(buf244, (16, 1), (1, 1), 0); del buf244  # reuse
        # Topologically Sorted Source Nodes: [e_17], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf246)
        buf247 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_34], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf246, buf247, 16, grid=grid(16), stream=stream0)
        buf248 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_34], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf247, buf248, 16, grid=grid(16), stream=stream0)
        buf249 = reinterpret_tensor(buf247, (4, 1, 4), (4, 4, 1), 0); del buf247  # reuse
        # Topologically Sorted Source Nodes: [bmm_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf249)
        buf250 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_17], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf249, buf243, buf250, 32, grid=grid(32), stream=stream0)
        buf251 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf250, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf251)
        buf252 = buf236; del buf236  # reuse
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf239, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf252)
        # Topologically Sorted Source Nodes: [ret_17], Original ATen: [aten._thnn_fused_gru_cell]
        buf253 = torch.ops.aten._thnn_fused_gru_cell.default(buf251, buf252, buf239, primals_8, primals_9)
        buf254 = buf253[0]
        buf256 = reinterpret_tensor(buf249, (4, 4), (4, 1), 0); del buf249  # reuse
        # Topologically Sorted Source Nodes: [probs_step_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf254, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf256)
        buf257 = buf243; del buf243  # reuse
        # Topologically Sorted Source Nodes: [next_input_17], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf256, buf257, 4, grid=grid(4), stream=stream0)
        buf258 = reinterpret_tensor(buf248, (4, 4), (4, 1), 0); del buf248  # reuse
        # Topologically Sorted Source Nodes: [linear_73], Original ATen: [aten.addmm]
        extern_kernels.mm(buf254, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf258)
        buf259 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_36, res_37], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf258, primals_4, buf259, 64, grid=grid(64), stream=stream0)
        buf260 = reinterpret_tensor(buf258, (16, 1), (1, 1), 0); del buf258  # reuse
        # Topologically Sorted Source Nodes: [e_18], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf260)
        buf261 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_36], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf260, buf261, 16, grid=grid(16), stream=stream0)
        buf262 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_36], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf261, buf262, 16, grid=grid(16), stream=stream0)
        buf263 = reinterpret_tensor(buf261, (4, 1, 4), (4, 4, 1), 0); del buf261  # reuse
        # Topologically Sorted Source Nodes: [bmm_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf263)
        buf264 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_18], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf263, buf257, buf264, 32, grid=grid(32), stream=stream0)
        buf265 = buf252; del buf252  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf264, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf265)
        buf266 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf254, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf266)
        # Topologically Sorted Source Nodes: [ret_18], Original ATen: [aten._thnn_fused_gru_cell]
        buf267 = torch.ops.aten._thnn_fused_gru_cell.default(buf265, buf266, buf254, primals_8, primals_9)
        buf268 = buf267[0]
        buf270 = reinterpret_tensor(buf263, (4, 4), (4, 1), 0); del buf263  # reuse
        # Topologically Sorted Source Nodes: [probs_step_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf268, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf270)
        buf271 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [next_input_18], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf270, buf271, 4, grid=grid(4), stream=stream0)
        buf272 = reinterpret_tensor(buf262, (4, 4), (4, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [linear_77], Original ATen: [aten.addmm]
        extern_kernels.mm(buf268, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf272)
        buf273 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_38, res_39], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf272, primals_4, buf273, 64, grid=grid(64), stream=stream0)
        buf274 = reinterpret_tensor(buf272, (16, 1), (1, 1), 0); del buf272  # reuse
        # Topologically Sorted Source Nodes: [e_19], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf274)
        buf275 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_38], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf274, buf275, 16, grid=grid(16), stream=stream0)
        buf276 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_38], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf275, buf276, 16, grid=grid(16), stream=stream0)
        buf277 = reinterpret_tensor(buf275, (4, 1, 4), (4, 4, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [bmm_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf276, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf277)
        buf278 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_19], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf277, buf271, buf278, 32, grid=grid(32), stream=stream0)
        buf279 = buf266; del buf266  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf278, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf279)
        buf280 = buf265; del buf265  # reuse
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten.mm]
        extern_kernels.mm(buf268, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf280)
        # Topologically Sorted Source Nodes: [ret_19], Original ATen: [aten._thnn_fused_gru_cell]
        buf281 = torch.ops.aten._thnn_fused_gru_cell.default(buf279, buf280, buf268, primals_8, primals_9)
        buf282 = buf281[0]
        buf284 = reinterpret_tensor(buf277, (4, 4), (4, 1), 0); del buf277  # reuse
        # Topologically Sorted Source Nodes: [probs_step_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf282, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf284)
        buf285 = buf271; del buf271  # reuse
        # Topologically Sorted Source Nodes: [next_input_19], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf284, buf285, 4, grid=grid(4), stream=stream0)
        buf286 = reinterpret_tensor(buf276, (4, 4), (4, 1), 0); del buf276  # reuse
        # Topologically Sorted Source Nodes: [linear_81], Original ATen: [aten.addmm]
        extern_kernels.mm(buf282, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf286)
        buf287 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_40, res_41], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf286, primals_4, buf287, 64, grid=grid(64), stream=stream0)
        buf288 = reinterpret_tensor(buf286, (16, 1), (1, 1), 0); del buf286  # reuse
        # Topologically Sorted Source Nodes: [e_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf288)
        buf289 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_40], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf288, buf289, 16, grid=grid(16), stream=stream0)
        buf290 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_40], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf289, buf290, 16, grid=grid(16), stream=stream0)
        buf291 = reinterpret_tensor(buf289, (4, 1, 4), (4, 4, 1), 0); del buf289  # reuse
        # Topologically Sorted Source Nodes: [bmm_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf290, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf291)
        buf292 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_20], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf291, buf285, buf292, 32, grid=grid(32), stream=stream0)
        buf293 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf292, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf293)
        buf294 = buf279; del buf279  # reuse
        # Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf282, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf294)
        # Topologically Sorted Source Nodes: [ret_20], Original ATen: [aten._thnn_fused_gru_cell]
        buf295 = torch.ops.aten._thnn_fused_gru_cell.default(buf293, buf294, buf282, primals_8, primals_9)
        buf296 = buf295[0]
        buf298 = reinterpret_tensor(buf291, (4, 4), (4, 1), 0); del buf291  # reuse
        # Topologically Sorted Source Nodes: [probs_step_20], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf296, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf298)
        buf300 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [next_input_20], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf298, buf300, 4, grid=grid(4), stream=stream0)
        buf301 = reinterpret_tensor(buf290, (4, 4), (4, 1), 0); del buf290  # reuse
        # Topologically Sorted Source Nodes: [linear_85], Original ATen: [aten.addmm]
        extern_kernels.mm(buf296, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf301)
        buf302 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_42, res_43], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf301, primals_4, buf302, 64, grid=grid(64), stream=stream0)
        buf303 = reinterpret_tensor(buf301, (16, 1), (1, 1), 0); del buf301  # reuse
        # Topologically Sorted Source Nodes: [e_21], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf302, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf303)
        buf304 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_42], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf303, buf304, 16, grid=grid(16), stream=stream0)
        buf305 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_42], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf304, buf305, 16, grid=grid(16), stream=stream0)
        buf306 = reinterpret_tensor(buf304, (4, 1, 4), (4, 4, 1), 0); del buf304  # reuse
        # Topologically Sorted Source Nodes: [bmm_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf305, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf306)
        buf307 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_21], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf306, buf300, buf307, 32, grid=grid(32), stream=stream0)
        buf308 = buf294; del buf294  # reuse
        # Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf307, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf308)
        buf309 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten.mm]
        extern_kernels.mm(buf296, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf309)
        # Topologically Sorted Source Nodes: [ret_21], Original ATen: [aten._thnn_fused_gru_cell]
        buf310 = torch.ops.aten._thnn_fused_gru_cell.default(buf308, buf309, buf296, primals_8, primals_9)
        buf311 = buf310[0]
        buf313 = reinterpret_tensor(buf306, (4, 4), (4, 1), 0); del buf306  # reuse
        # Topologically Sorted Source Nodes: [probs_step_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf311, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf313)
        buf314 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [next_input_21], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf313, buf314, 4, grid=grid(4), stream=stream0)
        buf315 = reinterpret_tensor(buf305, (4, 4), (4, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.addmm]
        extern_kernels.mm(buf311, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf315)
        buf316 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_44, res_45], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf315, primals_4, buf316, 64, grid=grid(64), stream=stream0)
        buf317 = reinterpret_tensor(buf315, (16, 1), (1, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [e_22], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf317)
        buf318 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_44], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf317, buf318, 16, grid=grid(16), stream=stream0)
        buf319 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_44], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf318, buf319, 16, grid=grid(16), stream=stream0)
        buf320 = reinterpret_tensor(buf318, (4, 1, 4), (4, 4, 1), 0); del buf318  # reuse
        # Topologically Sorted Source Nodes: [bmm_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf319, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf320)
        buf321 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_22], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf320, buf314, buf321, 32, grid=grid(32), stream=stream0)
        buf322 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf321, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf322)
        buf323 = buf308; del buf308  # reuse
        # Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf311, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf323)
        # Topologically Sorted Source Nodes: [ret_22], Original ATen: [aten._thnn_fused_gru_cell]
        buf324 = torch.ops.aten._thnn_fused_gru_cell.default(buf322, buf323, buf311, primals_8, primals_9)
        buf325 = buf324[0]
        buf327 = reinterpret_tensor(buf320, (4, 4), (4, 1), 0); del buf320  # reuse
        # Topologically Sorted Source Nodes: [probs_step_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf325, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf327)
        buf328 = buf314; del buf314  # reuse
        # Topologically Sorted Source Nodes: [next_input_22], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf327, buf328, 4, grid=grid(4), stream=stream0)
        buf329 = reinterpret_tensor(buf319, (4, 4), (4, 1), 0); del buf319  # reuse
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.addmm]
        extern_kernels.mm(buf325, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf329)
        buf330 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [res_46, res_47], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_1.run(buf1, buf329, primals_4, buf330, 64, grid=grid(64), stream=stream0)
        buf331 = reinterpret_tensor(buf329, (16, 1), (1, 1), 0); del buf329  # reuse
        # Topologically Sorted Source Nodes: [e_23], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf330, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf331)
        buf332 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_46], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf331, buf332, 16, grid=grid(16), stream=stream0)
        buf333 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_46], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf332, buf333, 16, grid=grid(16), stream=stream0)
        buf334 = reinterpret_tensor(buf332, (4, 1, 4), (4, 4, 1), 0); del buf332  # reuse
        # Topologically Sorted Source Nodes: [bmm_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf333, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf334)
        buf335 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_23], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf334, buf328, buf335, 32, grid=grid(32), stream=stream0)
        buf336 = buf323; del buf323  # reuse
        # Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf335, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf336)
        buf337 = buf322; del buf322  # reuse
        # Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten.mm]
        extern_kernels.mm(buf325, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf337)
        # Topologically Sorted Source Nodes: [ret_23], Original ATen: [aten._thnn_fused_gru_cell]
        buf338 = torch.ops.aten._thnn_fused_gru_cell.default(buf336, buf337, buf325, primals_8, primals_9)
        buf339 = buf338[0]
        buf341 = reinterpret_tensor(buf334, (4, 4), (4, 1), 0); del buf334  # reuse
        # Topologically Sorted Source Nodes: [probs_step_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf339, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf341)
        buf342 = buf328; del buf328  # reuse
        # Topologically Sorted Source Nodes: [next_input_23], Original ATen: [aten.argmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmax_5.run(buf341, buf342, 4, grid=grid(4), stream=stream0)
        buf343 = reinterpret_tensor(buf333, (4, 4), (4, 1), 0); del buf333  # reuse
        # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.addmm]
        extern_kernels.mm(buf339, reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf343)
        buf344 = reinterpret_tensor(buf1, (4, 4, 4), (16, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [res_48, res_49], Original ATen: [aten.add, aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_tanh_7.run(buf344, buf343, primals_4, 64, grid=grid(64), stream=stream0)
        del primals_4
        buf345 = reinterpret_tensor(buf343, (16, 1), (1, 1), 0); del buf343  # reuse
        # Topologically Sorted Source Nodes: [e_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf344, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 1), (1, 4), 0), out=buf345)
        buf346 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_48], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf345, buf346, 16, grid=grid(16), stream=stream0)
        buf347 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [alpha_48], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf346, buf347, 16, grid=grid(16), stream=stream0)
        buf348 = reinterpret_tensor(buf346, (4, 1, 4), (4, 4, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [bmm_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf347, (4, 1, 4), (4, 0, 1), 0), primals_1, out=buf348)
        del buf347
        buf349 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [concat_context_24], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf348, buf342, buf349, 32, grid=grid(32), stream=stream0)
        del buf342
        buf350 = buf337; del buf337  # reuse
        # Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf349, reinterpret_tensor(primals_6, (8, 12), (1, 8), 0), out=buf350)
        buf351 = buf336; del buf336  # reuse
        # Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf339, reinterpret_tensor(primals_7, (4, 12), (1, 4), 0), out=buf351)
        # Topologically Sorted Source Nodes: [ret_24], Original ATen: [aten._thnn_fused_gru_cell]
        buf352 = torch.ops.aten._thnn_fused_gru_cell.default(buf350, buf351, buf339, primals_8, primals_9)
        del buf350
        del buf351
        del primals_8
        del primals_9
        buf353 = buf352[0]
        buf355 = reinterpret_tensor(buf348, (4, 4), (4, 1), 0); del buf348  # reuse
        # Topologically Sorted Source Nodes: [probs_step_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_11, buf353, reinterpret_tensor(primals_10, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf355)
        del primals_11
        buf71 = empty_strided_cuda((4, 25, 4), (100, 4, 1), torch.float32)
        buf128 = buf71; del buf71  # reuse
        buf185 = buf128; del buf128  # reuse
        buf242 = buf185; del buf185  # reuse
        buf299 = buf242; del buf242  # reuse
        buf356 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [probs, setitem, setitem_1, setitem_2, setitem_3, setitem_4, setitem_5, setitem_6, setitem_7, setitem_8, setitem_9, setitem_10, setitem_11, setitem_12, setitem_13, setitem_14, setitem_15, setitem_16, setitem_17, setitem_18, setitem_19, setitem_20, setitem_21, setitem_22, setitem_23, setitem_24], Original ATen: [aten.zeros, aten.copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused_copy_zeros_8.run(buf356, buf70, buf56, buf42, buf28, buf14, buf127, buf113, buf99, buf85, buf184, buf170, buf156, buf142, buf241, buf227, buf213, buf199, buf298, buf284, buf270, buf256, buf355, buf341, buf327, buf313, 400, grid=grid(400), stream=stream0)
        del buf113
        del buf127
        del buf14
        del buf142
        del buf156
        del buf170
        del buf184
        del buf199
        del buf213
        del buf227
        del buf241
        del buf256
        del buf270
        del buf28
        del buf284
        del buf298
        del buf313
        del buf327
        del buf341
        del buf355
        del buf42
        del buf56
        del buf70
        del buf85
        del buf99
        buf84 = buf82[1]
        del buf82
        buf98 = buf96[1]
        del buf96
        buf112 = buf110[1]
        del buf110
        buf126 = buf124[1]
        del buf124
        buf141 = buf139[1]
        del buf139
        buf155 = buf153[1]
        del buf153
        buf169 = buf167[1]
        del buf167
        buf183 = buf181[1]
        del buf181
        buf198 = buf196[1]
        del buf196
        buf212 = buf210[1]
        del buf210
        buf226 = buf224[1]
        del buf224
        buf240 = buf238[1]
        del buf238
        buf255 = buf253[1]
        del buf253
        buf269 = buf267[1]
        del buf267
        buf283 = buf281[1]
        del buf281
        buf297 = buf295[1]
        del buf295
        buf312 = buf310[1]
        del buf310
        buf326 = buf324[1]
        del buf324
        buf340 = buf338[1]
        del buf338
        buf354 = buf352[1]
        del buf352
        buf357 = empty_strided_cuda((4, 25, 4), (100, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [probs_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_9.run(buf356, buf357, 400, grid=grid(400), stream=stream0)
        buf358 = buf356; del buf356  # reuse
        # Topologically Sorted Source Nodes: [probs_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_10.run(buf357, buf358, 400, grid=grid(400), stream=stream0)
        del buf357
    return (buf358, primals_1, buf0, buf3, buf4, buf8, buf12, buf13, buf17, buf18, buf22, buf26, buf27, buf31, buf32, buf36, buf40, buf41, buf45, buf46, buf50, buf54, buf55, buf59, buf60, buf64, buf68, buf69, buf74, buf75, buf79, buf83, buf84, buf88, buf89, buf93, buf97, buf98, buf102, buf103, buf107, buf111, buf112, buf116, buf117, buf121, buf125, buf126, buf131, buf132, buf136, buf140, buf141, buf145, buf146, buf150, buf154, buf155, buf159, buf160, buf164, buf168, buf169, buf173, buf174, buf178, buf182, buf183, buf188, buf189, buf193, buf197, buf198, buf202, buf203, buf207, buf211, buf212, buf216, buf217, buf221, buf225, buf226, buf230, buf231, buf235, buf239, buf240, buf245, buf246, buf250, buf254, buf255, buf259, buf260, buf264, buf268, buf269, buf273, buf274, buf278, buf282, buf283, buf287, buf288, buf292, buf296, buf297, buf302, buf303, buf307, buf311, buf312, buf316, buf317, buf321, buf325, buf326, buf330, buf331, buf335, buf339, buf340, buf344, buf345, buf349, buf353, buf354, buf358, primals_10, primals_7, primals_6, primals_5, primals_3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((12, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((12, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
