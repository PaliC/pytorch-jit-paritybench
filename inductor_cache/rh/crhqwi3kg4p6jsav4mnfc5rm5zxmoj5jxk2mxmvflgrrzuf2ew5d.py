# AOT ID: ['0_forward']
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


# kernel path: inductor_cache/fb/cfbrrfntfocdlxs5xgmpyczbkdzwj7rysjx4vxxyp4zuric6rtq4.py
# Topologically Sorted Source Nodes: [_align], Original ATen: [aten.clone, aten._unsafe_view]
# Source node to ATen node mapping:
#   _align => clone, view
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [16, 4]), kwargs = {})
triton_poi_fused__unsafe_view_clone_0 = async_compile.triton('triton_poi_fused__unsafe_view_clone_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_view_clone_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (4*x1 + 16*(y0 // 4) + ((y0 % 4))), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qd/cqdmibl6viy37xawl2f7fdevwil6xzaaj4tpmmgmpetasi2thdgc.py
# Topologically Sorted Source Nodes: [_align_2, max_1, max_2, _scoreA_1, _scoreB_1], Original ATen: [aten.tanh, aten.max, aten._softmax]
# Source node to ATen node mapping:
#   _align_2 => tanh
#   _scoreA_1 => amax
#   _scoreB_1 => amax_1
#   max_1 => max_1
#   max_2 => max_2
# Graph fragment:
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%bmm,), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%tanh, 2), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%tanh, 1), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%getitem, [-1], True), kwargs = {})
#   %amax_1 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%getitem_2, [-1], True), kwargs = {})
triton_poi_fused__softmax_max_tanh_1 = async_compile.triton('triton_poi_fused__softmax_max_tanh_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_max_tanh_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_max_tanh_1(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = libdevice.tanh(tmp11)
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = triton_helpers.maximum(tmp12, tmp14)
    tmp17 = libdevice.tanh(tmp16)
    tmp18 = triton_helpers.maximum(tmp15, tmp17)
    tmp20 = libdevice.tanh(tmp19)
    tmp21 = triton_helpers.maximum(tmp18, tmp20)
    tmp22 = triton_helpers.maximum(tmp10, tmp21)
    tmp24 = libdevice.tanh(tmp23)
    tmp26 = libdevice.tanh(tmp25)
    tmp27 = triton_helpers.maximum(tmp24, tmp26)
    tmp29 = libdevice.tanh(tmp28)
    tmp30 = triton_helpers.maximum(tmp27, tmp29)
    tmp32 = libdevice.tanh(tmp31)
    tmp33 = triton_helpers.maximum(tmp30, tmp32)
    tmp34 = triton_helpers.maximum(tmp22, tmp33)
    tmp36 = libdevice.tanh(tmp35)
    tmp38 = libdevice.tanh(tmp37)
    tmp39 = triton_helpers.maximum(tmp36, tmp38)
    tmp41 = libdevice.tanh(tmp40)
    tmp42 = triton_helpers.maximum(tmp39, tmp41)
    tmp44 = libdevice.tanh(tmp43)
    tmp45 = triton_helpers.maximum(tmp42, tmp44)
    tmp46 = triton_helpers.maximum(tmp34, tmp45)
    tmp47 = triton_helpers.maximum(tmp1, tmp12)
    tmp48 = triton_helpers.maximum(tmp47, tmp24)
    tmp49 = triton_helpers.maximum(tmp48, tmp36)
    tmp50 = triton_helpers.maximum(tmp3, tmp14)
    tmp51 = triton_helpers.maximum(tmp50, tmp26)
    tmp52 = triton_helpers.maximum(tmp51, tmp38)
    tmp53 = triton_helpers.maximum(tmp49, tmp52)
    tmp54 = triton_helpers.maximum(tmp6, tmp17)
    tmp55 = triton_helpers.maximum(tmp54, tmp29)
    tmp56 = triton_helpers.maximum(tmp55, tmp41)
    tmp57 = triton_helpers.maximum(tmp53, tmp56)
    tmp58 = triton_helpers.maximum(tmp9, tmp20)
    tmp59 = triton_helpers.maximum(tmp58, tmp32)
    tmp60 = triton_helpers.maximum(tmp59, tmp44)
    tmp61 = triton_helpers.maximum(tmp57, tmp60)
    tl.store(out_ptr0 + (x0), tmp46, xmask)
    tl.store(out_ptr1 + (x0), tmp61, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/el/celo2kkefsmqq6q5hngz5lkt6egweu4bobtsa5wsc6dofeo3nkhm.py
# Topologically Sorted Source Nodes: [_align_2, max_1, _scoreA_1], Original ATen: [aten.tanh, aten.max, aten._softmax]
# Source node to ATen node mapping:
#   _align_2 => tanh
#   _scoreA_1 => exp, sub
#   max_1 => max_1
# Graph fragment:
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%bmm,), kwargs = {})
#   %max_1 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%tanh, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_max_tanh_2 = async_compile.triton('triton_poi_fused__softmax_max_tanh_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_max_tanh_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_max_tanh_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp10 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5skf3f5sqvrvoj3e2emovhzkxskglss7me4yw46p4hq3iv6gm76.py
# Topologically Sorted Source Nodes: [_align_2, max_2, _scoreB_1], Original ATen: [aten.tanh, aten.max, aten._softmax]
# Source node to ATen node mapping:
#   _align_2 => tanh
#   _scoreB_1 => exp_1, sub_1
#   max_2 => max_2
# Graph fragment:
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%bmm,), kwargs = {})
#   %max_2 : [num_users=1] = call_function[target=torch.ops.aten.max.dim](args = (%tanh, 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%getitem_2, %amax_1), kwargs = {})
#   %exp_1 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
triton_poi_fused__softmax_max_tanh_3 = async_compile.triton('triton_poi_fused__softmax_max_tanh_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_max_tanh_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_max_tanh_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp2 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp5 = tl.load(in_ptr0 + (8 + x0 + 16*x1), xmask)
    tmp8 = tl.load(in_ptr0 + (12 + x0 + 16*x1), xmask)
    tmp11 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = triton_helpers.maximum(tmp1, tmp3)
    tmp6 = libdevice.tanh(tmp5)
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = libdevice.tanh(tmp8)
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp10 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxeercglymjqip4jf3xl5kzkf73hf4shg7i7tr4dl3lod6ymhsr.py
# Topologically Sorted Source Nodes: [_scoreA_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   _scoreA_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_4 = async_compile.triton('triton_poi_fused__softmax_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_align], Original ATen: [aten.clone, aten._unsafe_view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_view_clone_0.run(primals_1, buf0, 16, 4, grid=grid(16, 4), stream=stream0)
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_align], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, primals_3, out=buf1)
        del primals_3
        buf2 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_align_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1, (4, 4, 4), (16, 4, 1), 0), primals_2, out=buf2)
        del buf1
        buf3 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        buf5 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [_align_2, max_1, max_2, _scoreA_1, _scoreB_1], Original ATen: [aten.tanh, aten.max, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_max_tanh_1.run(buf2, buf3, buf5, 4, grid=grid(4), stream=stream0)
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_align_2, max_1, _scoreA_1], Original ATen: [aten.tanh, aten.max, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_max_tanh_2.run(buf2, buf3, buf4, 16, grid=grid(16), stream=stream0)
        del buf3
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_align_2, max_2, _scoreB_1], Original ATen: [aten.tanh, aten.max, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_max_tanh_3.run(buf2, buf5, buf6, 16, grid=grid(16), stream=stream0)
        del buf5
        buf7 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [_scoreA_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf4, buf7, 16, grid=grid(16), stream=stream0)
        buf8 = reinterpret_tensor(buf4, (4, 4, 1), (4, 1, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(primals_1, reinterpret_tensor(buf7, (4, 4, 1), (4, 1, 0), 0), out=buf8)
        buf9 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [_scoreB_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf6, buf9, 16, grid=grid(16), stream=stream0)
        buf10 = reinterpret_tensor(buf6, (4, 4, 1), (4, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(primals_2, reinterpret_tensor(buf9, (4, 4, 1), (4, 1, 0), 0), out=buf10)
        del buf9
    return (reinterpret_tensor(buf8, (4, 4), (4, 1), 0), reinterpret_tensor(buf10, (4, 4), (4, 1), 0), buf2, reinterpret_tensor(primals_2, (4, 4, 4), (16, 1, 4), 0), reinterpret_tensor(primals_1, (4, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf0, (4, 16), (1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
