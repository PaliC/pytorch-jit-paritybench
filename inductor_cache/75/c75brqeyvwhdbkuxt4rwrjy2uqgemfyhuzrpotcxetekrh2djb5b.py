# AOT ID: ['10_forward']
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


# kernel path: inductor_cache/eu/ceu7upc5sgbuphhxv6k4aea2igswmsev5w6afoagdy5xvecvkkn5.py
# Topologically Sorted Source Nodes: [Q_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   Q_1 => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_2, 1.0), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/73/c73ig4rbudlerzep2og4zez73x3rsxgq2yexmlbfwiwcptw2ztr6.py
# Topologically Sorted Source Nodes: [logits], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   logits => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dn/cdnh4oiaxklbq73i267sm7wsfxjol3ndbfnrbmubngbnbm54sb6n.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + 32*x1), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4g/c4gewrjketnkm4oyk2vafdqqqbfrdoxc2kvf72wkbs5sdoxhb5i4.py
# Topologically Sorted Source Nodes: [mul, sub, mul_1, logits_1, weights], Original ATen: [aten.mul, aten.rsub, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   logits_1 => add
#   mul => mul_1
#   mul_1 => mul_2
#   sub => sub
#   weights => amax, exp, sub_1, sum_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %view_12), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %view_12), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -1e+30), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_poi_fused__softmax_add_mul_rsub_3 = async_compile.triton('triton_poi_fused__softmax_add_mul_rsub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_mul_rsub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_mul_rsub_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (4*x2), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + 4*x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (3 + 4*x2), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 - tmp1
    tmp5 = -1e+30
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 + tmp6
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 - tmp9
    tmp12 = tmp11 * tmp5
    tmp13 = tmp10 + tmp12
    tmp14 = triton_helpers.maximum(tmp7, tmp13)
    tmp17 = tmp15 * tmp16
    tmp18 = tmp3 - tmp16
    tmp19 = tmp18 * tmp5
    tmp20 = tmp17 + tmp19
    tmp21 = triton_helpers.maximum(tmp14, tmp20)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp3 - tmp23
    tmp26 = tmp25 * tmp5
    tmp27 = tmp24 + tmp26
    tmp28 = triton_helpers.maximum(tmp21, tmp27)
    tmp29 = tmp7 - tmp28
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp13 - tmp28
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp30 + tmp32
    tmp34 = tmp20 - tmp28
    tmp35 = tl_math.exp(tmp34)
    tmp36 = tmp33 + tmp35
    tmp37 = tmp27 - tmp28
    tmp38 = tl_math.exp(tmp37)
    tmp39 = tmp36 + tmp38
    tl.store(out_ptr0 + (x2), tmp28, xmask)
    tl.store(out_ptr1 + (x2), tmp39, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lg/clg2luhielc6fqouaem5ngls3whrium2qd54jq3jzgp2x3ultqhu.py
# Topologically Sorted Source Nodes: [mul, sub, mul_1, logits_1, weights], Original ATen: [aten.mul, aten.rsub, aten.add, aten._softmax]
# Source node to ATen node mapping:
#   logits_1 => add
#   mul => mul_1
#   mul_1 => mul_2
#   sub => sub
#   weights => amax, div, exp, sub_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, %view_12), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %view_12), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, -1e+30), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %mul_2), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%add, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_add_mul_rsub_4 = async_compile.triton('triton_poi_fused__softmax_add_mul_rsub_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_mul_rsub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_mul_rsub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 64
    x4 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 - tmp1
    tmp5 = -1e+30
    tmp6 = tmp4 * tmp5
    tmp7 = tmp2 + tmp6
    tmp9 = tmp7 - tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp12 = tmp10 / tmp11
    tl.store(in_out_ptr0 + (x3), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qv/cqvvputffj3l6llgfyakx7knd4rkyc2cnhaw6be6curiqquqito6.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_15,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (8, 4, 1), (4, 1, 1))
    assert_size_stride(primals_3, (4, 4, 1), (4, 1, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [query], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(primals_1, primals_3, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 4), (16, 4, 1))
        buf2 = reinterpret_tensor(buf1, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [Q_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(buf2, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [memory], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf0, (4, 8, 4), (32, 4, 1))
        buf3 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf0, buf3, 64, grid=grid(64), stream=stream0)
        buf8 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf0, buf8, 64, grid=grid(64), stream=stream0)
        del buf0
        buf4 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [logits], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf2, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf3, (16, 1, 4), (4, 0, 1), 0), out=buf4)
        buf5 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        buf6 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [mul, sub, mul_1, logits_1, weights], Original ATen: [aten.mul, aten.rsub, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_mul_rsub_3.run(buf4, primals_4, buf5, buf6, 64, grid=grid(64), stream=stream0)
        buf7 = reinterpret_tensor(buf4, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [mul, sub, mul_1, logits_1, weights], Original ATen: [aten.mul, aten.rsub, aten.add, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_mul_rsub_4.run(buf7, primals_4, buf5, buf6, 256, grid=grid(256), stream=stream0)
        buf9 = reinterpret_tensor(buf6, (16, 4, 1), (4, 1, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf8, (16, 4, 1), (4, 1, 0), 0), out=buf9)
        buf10 = reinterpret_tensor(buf5, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf9, buf10, 16, 4, grid=grid(16, 4), stream=stream0)
        del buf9
    return (reinterpret_tensor(buf10, (4, 4, 4), (16, 1, 4), 0), primals_1, primals_2, primals_3, primals_4, buf7, reinterpret_tensor(buf8, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf2, (16, 1, 4), (4, 4, 1), 0), reinterpret_tensor(buf3, (16, 4, 1), (4, 1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((8, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
