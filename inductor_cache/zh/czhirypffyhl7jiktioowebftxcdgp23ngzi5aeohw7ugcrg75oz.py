# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/x6/cx6vqb7wt5krfko2xy7lrqwdan2ybs3o3h46gqsdewl3rfswlblg.py
# Topologically Sorted Source Nodes: [e, e_1], Original ATen: [aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   e => add
#   e_1 => gt
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %permute), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add, 0), kwargs = {})
triton_poi_fused_add_leaky_relu_0 = async_compile.triton('triton_poi_fused_add_leaky_relu_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x0 = (xindex % 4)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tl.store(out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eg/cegj7u6mvkniowhhvxvp3ea62orr6wmle4en3dvjxcsigl2rkett.py
# Topologically Sorted Source Nodes: [gt], Original ATen: [aten.gt]
# Source node to ATen node mapping:
#   gt => gt_1
# Graph fragment:
#   %gt_1 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_4, 0), kwargs = {})
triton_poi_fused_gt_1 = async_compile.triton('triton_poi_fused_gt_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gt_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gt_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2g/c2gjxydbhjfrjrkw5yqwqhlgkvg4omuzm5myn4hwfdkoy7fp74qk.py
# Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
# Source node to ATen node mapping:
#   attention => where_1
#   attention_1 => amax
#   e => add
#   e_1 => mul, where
#   zero_vec => full_default
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %permute), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 4), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add, %mul), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], -8999999815811072.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where, %full_default), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where_1, [1], True), kwargs = {})
triton_poi_fused__softmax_add_leaky_relu_mul_where_2 = async_compile.triton('triton_poi_fused__softmax_add_leaky_relu_mul_where_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_leaky_relu_mul_where_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_leaky_relu_mul_where_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x0), xmask)
    tmp3 = tl.load(in_ptr3 + (0))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp12 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp13 = tl.load(in_ptr3 + (1))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp21 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp22 = tl.load(in_ptr3 + (2))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp29 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp30 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp31 = tl.load(in_ptr3 + (3))
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK])
    tmp5 = tmp2 + tmp4
    tmp6 = 4.0
    tmp7 = tmp5 * tmp6
    tmp8 = tl.where(tmp1, tmp5, tmp7)
    tmp9 = -8999999815811072.0
    tmp10 = tl.where(tmp0, tmp8, tmp9)
    tmp15 = tmp2 + tmp14
    tmp16 = tmp15 * tmp6
    tmp17 = tl.where(tmp12, tmp15, tmp16)
    tmp18 = tl.where(tmp11, tmp17, tmp9)
    tmp19 = triton_helpers.maximum(tmp10, tmp18)
    tmp24 = tmp2 + tmp23
    tmp25 = tmp24 * tmp6
    tmp26 = tl.where(tmp21, tmp24, tmp25)
    tmp27 = tl.where(tmp20, tmp26, tmp9)
    tmp28 = triton_helpers.maximum(tmp19, tmp27)
    tmp33 = tmp2 + tmp32
    tmp34 = tmp33 * tmp6
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp29, tmp35, tmp9)
    tmp37 = triton_helpers.maximum(tmp28, tmp36)
    tl.store(out_ptr0 + (x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qb/cqbzxzbfniomp6m4s7gvsxycsjnmn6v5bxnfkrsix7hdpmcuqryy.py
# Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
# Source node to ATen node mapping:
#   attention => where_1
#   attention_1 => exp, sub
#   e => add
#   e_1 => mul, where
#   zero_vec => full_default
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_1, %permute), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 4), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add, %mul), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], -8999999815811072.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt_1, %where, %full_default), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_add_leaky_relu_mul_where_3 = async_compile.triton('triton_poi_fused__softmax_add_leaky_relu_mul_where_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*i1', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_add_leaky_relu_mul_where_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_add_leaky_relu_mul_where_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.int1)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp4 = tmp2 + tmp3
    tmp5 = 4.0
    tmp6 = tmp4 * tmp5
    tmp7 = tl.where(tmp1, tmp4, tmp6)
    tmp8 = -8999999815811072.0
    tmp9 = tl.where(tmp0, tmp7, tmp8)
    tmp11 = tmp9 - tmp10
    tmp12 = tl_math.exp(tmp11)
    tl.store(out_ptr0 + (x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxeercglymjqip4jf3xl5kzkf73hf4shg7i7tr4dl3lod6ymhsr.py
# Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attention_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
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


# kernel path: inductor_cache/ag/cagyw46ggj6hriffcsdjlgkcntfmkfesth267oq7krd2so4g3524.py
# Topologically Sorted Source Nodes: [elu], Original ATen: [aten.elu]
# Source node to ATen node mapping:
#   elu => expm1, gt_2, mul_2, mul_4, where_2
# Graph fragment:
#   %gt_2 : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%mm_3, 0), kwargs = {})
#   %mul_2 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm_3, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul_2,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_2, %mul_2, %mul_4), kwargs = {})
triton_poi_fused_elu_5 = async_compile.triton('triton_poi_fused_elu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_elu_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 1.0
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.expm1(tmp4)
    tmp6 = tmp5 * tmp3
    tmp7 = tl.where(tmp2, tmp4, tmp6)
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (8, 1), (1, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh], Original ATen: [aten.mm]
        extern_kernels.mm(primals_2, primals_1, out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh1], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (4, 1), (1, 1), 0), out=buf1)
        buf2 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Wh2], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(primals_3, (4, 1), (1, 1), 4), out=buf2)
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [e, e_1], Original ATen: [aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_0.run(buf1, buf2, buf3, 16, grid=grid(16), stream=stream0)
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [gt], Original ATen: [aten.gt]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gt_1.run(primals_4, buf4, 16, grid=grid(16), stream=stream0)
        del primals_4
        buf5 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_leaky_relu_mul_where_2.run(buf4, buf3, buf1, buf2, buf5, 4, grid=grid(4), stream=stream0)
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [e, e_1, zero_vec, attention, attention_1], Original ATen: [aten.add, aten.leaky_relu, aten.mul, aten.where, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_add_leaky_relu_mul_where_3.run(buf4, buf3, buf1, buf2, buf5, buf6, 16, grid=grid(16), stream=stream0)
        del buf1
        del buf2
        del buf5
        buf7 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf6, buf7, 16, grid=grid(16), stream=stream0)
        buf8 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [h_prime], Original ATen: [aten.mm]
        extern_kernels.mm(buf7, buf0, out=buf8)
        buf9 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [elu], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_5.run(buf8, buf9, 16, grid=grid(16), stream=stream0)
    return (buf9, buf3, buf4, buf7, buf8, reinterpret_tensor(buf0, (4, 4), (1, 4), 0), reinterpret_tensor(primals_3, (1, 4), (1, 1), 4), reinterpret_tensor(primals_3, (1, 4), (1, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((8, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
