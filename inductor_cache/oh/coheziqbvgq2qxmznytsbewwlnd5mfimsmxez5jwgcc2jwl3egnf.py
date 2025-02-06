# AOT ID: ['29_inference']
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


# kernel path: inductor_cache/6e/c6ea5mvmqz5zwjzjxmsvwjfyktvjzyybor22yvfyck76hrh5zclv.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   x_4 => _adaptive_avg_pool2d_1
# Graph fragment:
#   %_adaptive_avg_pool2d_1 : [num_users=4] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%arg0_1, [3, 3]), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_0 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 3) % 3)
    x0 = (xindex % 3)
    x2 = xindex // 9
    x4 = xindex
    tmp0 = (4*x1) // 3
    tmp1 = 2 + ((4*x1) // 3)
    tmp2 = tmp0 < tmp1
    tmp3 = (4*x0) // 3
    tmp4 = 2 + ((4*x0) // 3)
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (4*((4*x1) // 3) + 16*x2 + ((4*x0) // 3)), tmp6 & xmask, other=0.0)
    tmp8 = 1 + ((4*x0) // 3)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 4*((4*x1) // 3) + 16*x2 + ((4*x0) // 3)), tmp10 & xmask, other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 1 + ((4*x1) // 3)
    tmp14 = tmp13 < tmp1
    tmp15 = tmp14 & tmp5
    tmp16 = tl.load(in_ptr0 + (4 + 4*((4*x1) // 3) + 16*x2 + ((4*x0) // 3)), tmp15 & xmask, other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp14 & tmp9
    tmp19 = tl.load(in_ptr0 + (5 + 4*((4*x1) // 3) + 16*x2 + ((4*x0) // 3)), tmp18 & xmask, other=0.0)
    tmp20 = tmp19 + tmp17
    tmp21 = 1.0
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = 1.0
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tmp26 + tmp23
    tmp28 = 1.0
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp15, tmp28, tmp29)
    tmp31 = tmp30 + tmp27
    tmp32 = 1.0
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp18, tmp32, tmp33)
    tmp35 = tmp34 + tmp31
    tmp36 = tmp20 / tmp35
    tl.store(out_ptr0 + (x4), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3r/c3r73iyk64zkwnlvqrrxoxyunfgub4ofephvjtvnka4urxxkjkma.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   x_6 => _adaptive_avg_pool2d_2
# Graph fragment:
#   %_adaptive_avg_pool2d_2 : [num_users=4] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%arg0_1, [6, 6]), kwargs = {})
triton_poi_fused__adaptive_avg_pool2d_1 = async_compile.triton('triton_poi_fused__adaptive_avg_pool2d_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 6) % 6)
    x0 = (xindex % 6)
    x2 = xindex // 36
    x4 = xindex
    tmp0 = (2*x1) // 3
    tmp1 = (9 + 4*x1) // 6
    tmp2 = tmp0 < tmp1
    tmp3 = (2*x0) // 3
    tmp4 = (9 + 4*x0) // 6
    tmp5 = tmp3 < tmp4
    tmp6 = tmp2 & tmp5
    tmp7 = tl.load(in_ptr0 + (4*((2*x1) // 3) + 16*x2 + ((2*x0) // 3)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = 1 + ((2*x0) // 3)
    tmp9 = tmp8 < tmp4
    tmp10 = tmp2 & tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 4*((2*x1) // 3) + 16*x2 + ((2*x0) // 3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp11 + tmp7
    tmp13 = 1 + ((2*x1) // 3)
    tmp14 = tmp13 < tmp1
    tmp15 = tmp14 & tmp5
    tmp16 = tl.load(in_ptr0 + (4 + 4*((2*x1) // 3) + 16*x2 + ((2*x0) // 3)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp16 + tmp12
    tmp18 = tmp14 & tmp9
    tmp19 = tl.load(in_ptr0 + (5 + 4*((2*x1) // 3) + 16*x2 + ((2*x0) // 3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp19 + tmp17
    tmp21 = 1.0
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp6, tmp21, tmp22)
    tmp24 = 1.0
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp10, tmp24, tmp25)
    tmp27 = tmp26 + tmp23
    tmp28 = 1.0
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp15, tmp28, tmp29)
    tmp31 = tmp30 + tmp27
    tmp32 = 1.0
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp18, tmp32, tmp33)
    tmp35 = tmp34 + tmp31
    tmp36 = tmp20 / tmp35
    tl.store(out_ptr0 + (x4), tmp36, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k2/ck2iwrxsxkr35k7u6k3sdvtgrwxlpoy4q6b6qemwmrciu4cfx6hf.py
# Topologically Sorted Source Nodes: [x, x_1, feat, x_2, x_3, feat_1, x_5, feat_2, x_7, feat_3], Original ATen: [aten.mean, aten._to_copy, aten.arange, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add, aten._adaptive_avg_pool2d]
# Source node to ATen node mapping:
#   feat => add_5
#   feat_1 => add_11
#   feat_2 => add_17
#   feat_3 => add_23
#   x => mean
#   x_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_2, add_3, add_4, clamp_max_2, clamp_max_3, clamp_min_1, clamp_min_2, clamp_min_3, convert_element_type_1, convert_element_type_2, convert_element_type_3, iota_1, mul_1, mul_2, mul_3, mul_4, sub, sub_1, sub_2, sub_3, sub_4
#   x_2 => _adaptive_avg_pool2d
#   x_3 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_10, add_8, add_9, clamp_max_6, clamp_max_7, clamp_min_5, clamp_min_6, clamp_min_7, convert_element_type_5, convert_element_type_6, convert_element_type_7, iota_3, mul_6, mul_7, mul_8, mul_9, sub_5, sub_6, sub_7, sub_8, sub_9
#   x_5 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_14, add_15, add_16, clamp_max_10, clamp_max_11, clamp_min_10, clamp_min_11, clamp_min_9, convert_element_type_10, convert_element_type_11, convert_element_type_9, iota_5, mul_11, mul_12, mul_13, mul_14, sub_10, sub_11, sub_12, sub_13, sub_14
#   x_7 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_20, add_21, add_22, clamp_max_14, clamp_max_15, clamp_min_13, clamp_min_14, clamp_min_15, convert_element_type_13, convert_element_type_14, convert_element_type_15, iota_7, mul_16, mul_17, mul_18, mul_19, sub_15, sub_16, sub_17, sub_18, sub_19
# Graph fragment:
#   %mean : [num_users=4] = call_function[target=torch.ops.aten.mean.dim](args = (%arg0_1, [-1, -2], True), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view, torch.int64), kwargs = {})
#   %iota_1 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_1, torch.float32), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2, 0.0), kwargs = {})
#   %clamp_min_1 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_1, 0.0), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_1, torch.int64), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %clamp_max, %clamp_max_1]), kwargs = {})
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %clamp_max, %convert_element_type_3]), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_3, %_unsafe_index_2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_1, %convert_element_type_3), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 1.0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %clamp_max_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %mul_3), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %convert_element_type_1, %clamp_max_1]), kwargs = {})
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mean, [None, None, %convert_element_type_1, %convert_element_type_3]), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %clamp_max_2), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %add_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %convert_element_type_1), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 1.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %clamp_max_3), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %mul_4), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %add_4), kwargs = {})
#   %_adaptive_avg_pool2d : [num_users=4] = call_function[target=torch.ops.aten._adaptive_avg_pool2d.default](args = (%arg0_1, [2, 2]), kwargs = {})
#   %convert_element_type_5 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_3, torch.float32), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_6, 0.3333333333333333), kwargs = {})
#   %clamp_min_5 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_6, 0.0), kwargs = {})
#   %convert_element_type_7 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_5, torch.int64), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %clamp_max_4, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %clamp_max_4, %convert_element_type_7]), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_7, %_unsafe_index_6), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_5, %convert_element_type_7), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_5, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 1.0), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %clamp_max_6), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_6, %mul_8), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %convert_element_type_5, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d, [None, None, %convert_element_type_5, %convert_element_type_7]), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %clamp_max_6), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_7), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %add_8), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_2, %convert_element_type_5), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_8, 0.0), kwargs = {})
#   %clamp_max_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 1.0), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %clamp_max_7), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %mul_9), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %add_10), kwargs = {})
#   %convert_element_type_9 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
#   %iota_5 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_10 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_5, torch.float32), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_10, 0.6666666666666666), kwargs = {})
#   %clamp_min_9 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_11, 0.0), kwargs = {})
#   %convert_element_type_11 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_9, torch.int64), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_1, [None, None, %clamp_max_8, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_10 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_1, [None, None, %clamp_max_8, %convert_element_type_11]), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_11, %_unsafe_index_10), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_9, %convert_element_type_11), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_10, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 1.0), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %clamp_max_10), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_10, %mul_13), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_1, [None, None, %convert_element_type_9, %clamp_max_9]), kwargs = {})
#   %_unsafe_index_8 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_1, [None, None, %convert_element_type_9, %convert_element_type_11]), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_9, %_unsafe_index_8), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %clamp_max_10), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_8, %mul_12), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %add_14), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %convert_element_type_9), kwargs = {})
#   %clamp_min_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_13, 0.0), kwargs = {})
#   %clamp_max_11 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_11, 1.0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %clamp_max_11), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %mul_14), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %add_16), kwargs = {})
#   %convert_element_type_13 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.int64), kwargs = {})
#   %iota_7 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_7, torch.float32), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_14, 1.6666666666666667), kwargs = {})
#   %clamp_min_13 : [num_users=2] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_16, 0.0), kwargs = {})
#   %convert_element_type_15 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_min_13, torch.int64), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_2, [None, None, %clamp_max_12, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_14 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_2, [None, None, %clamp_max_12, %convert_element_type_15]), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_15, %_unsafe_index_14), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_13, %convert_element_type_15), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_15, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 1.0), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %clamp_max_14), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_14, %mul_18), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_2, [None, None, %convert_element_type_13, %clamp_max_13]), kwargs = {})
#   %_unsafe_index_12 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%_adaptive_avg_pool2d_2, [None, None, %convert_element_type_13, %convert_element_type_15]), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_13, %_unsafe_index_12), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %clamp_max_14), kwargs = {})
#   %add_20 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_12, %mul_17), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_21, %add_20), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_6, %convert_element_type_13), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_18, 0.0), kwargs = {})
#   %clamp_max_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 1.0), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %clamp_max_15), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %mul_19), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %add_22), kwargs = {})
triton_per_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_clamp_mean_mul_sub_2 = async_compile.triton('triton_per_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_clamp_mean_mul_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr4': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_clamp_mean_mul_sub_2', 'mutated_arg_names': ['in_out_ptr4'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_clamp_mean_mul_sub_2(in_out_ptr4, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    r3 = rindex // 4
    r2 = (rindex % 4)
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 16.0
    tmp6 = tmp4 / tmp5
    tmp7 = tmp6 - tmp6
    tmp8 = 0.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10 - tmp10
    tmp12 = tmp11 * tmp8
    tmp13 = tmp10 + tmp12
    tmp14 = tmp0 + tmp13
    tmp15 = r3
    tmp16 = tmp15.to(tl.float32)
    tmp17 = 0.3333333333333333
    tmp18 = tmp16 * tmp17
    tmp19 = triton_helpers.maximum(tmp18, tmp8)
    tmp20 = tmp19.to(tl.int32)
    tmp21 = tl.full([1, 1], 1, tl.int64)
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.minimum(tmp22, tmp21)
    tmp24 = r2
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp17
    tmp27 = triton_helpers.maximum(tmp26, tmp8)
    tmp28 = tmp27.to(tl.int32)
    tmp29 = tmp28 + tmp21
    tmp30 = triton_helpers.minimum(tmp29, tmp21)
    tmp31 = tl.load(in_ptr0 + (2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (1 + 2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp33 = tmp32 + tmp31
    tmp34 = tl.load(in_ptr0 + (4 + 2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp35 = tmp34 + tmp33
    tmp36 = tl.load(in_ptr0 + (5 + 2*tmp30 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp37 = tmp36 + tmp35
    tmp38 = 0.25
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr0 + (2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr0 + (1 + 2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp42 = tmp41 + tmp40
    tmp43 = tl.load(in_ptr0 + (4 + 2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp44 = tmp43 + tmp42
    tmp45 = tl.load(in_ptr0 + (5 + 2*tmp28 + 8*tmp23 + 16*x0), xmask, eviction_policy='evict_last')
    tmp46 = tmp45 + tmp44
    tmp47 = tmp46 * tmp38
    tmp48 = tmp39 - tmp47
    tmp49 = tl.load(in_ptr0 + (2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr0 + (1 + 2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp51 = tmp50 + tmp49
    tmp52 = tl.load(in_ptr0 + (4 + 2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp53 = tmp52 + tmp51
    tmp54 = tl.load(in_ptr0 + (5 + 2*tmp30 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp55 = tmp54 + tmp53
    tmp56 = tmp55 * tmp38
    tmp57 = tl.load(in_ptr0 + (2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr0 + (1 + 2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp59 = tmp58 + tmp57
    tmp60 = tl.load(in_ptr0 + (4 + 2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp61 = tmp60 + tmp59
    tmp62 = tl.load(in_ptr0 + (5 + 2*tmp28 + 8*tmp20 + 16*x0), xmask, eviction_policy='evict_last')
    tmp63 = tmp62 + tmp61
    tmp64 = tmp63 * tmp38
    tmp65 = tmp56 - tmp64
    tmp66 = tmp28.to(tl.float32)
    tmp67 = tmp27 - tmp66
    tmp68 = triton_helpers.maximum(tmp67, tmp8)
    tmp69 = 1.0
    tmp70 = triton_helpers.minimum(tmp68, tmp69)
    tmp71 = tmp48 * tmp70
    tmp72 = tmp47 + tmp71
    tmp73 = tmp65 * tmp70
    tmp74 = tmp64 + tmp73
    tmp75 = 0.6666666666666666
    tmp76 = tmp16 * tmp75
    tmp77 = triton_helpers.maximum(tmp76, tmp8)
    tmp78 = tmp77.to(tl.int32)
    tmp79 = tmp78 + tmp21
    tmp80 = tl.full([1, 1], 2, tl.int64)
    tmp81 = triton_helpers.minimum(tmp79, tmp80)
    tmp82 = tmp25 * tmp75
    tmp83 = triton_helpers.maximum(tmp82, tmp8)
    tmp84 = tmp83.to(tl.int32)
    tmp85 = tl.load(in_ptr1 + (tmp84 + 3*tmp81 + 9*x0), xmask, eviction_policy='evict_last')
    tmp86 = tmp84 + tmp21
    tmp87 = triton_helpers.minimum(tmp86, tmp80)
    tmp88 = tl.load(in_ptr1 + (tmp87 + 3*tmp81 + 9*x0), xmask, eviction_policy='evict_last')
    tmp89 = tmp88 - tmp85
    tmp90 = tmp84.to(tl.float32)
    tmp91 = tmp83 - tmp90
    tmp92 = triton_helpers.maximum(tmp91, tmp8)
    tmp93 = triton_helpers.minimum(tmp92, tmp69)
    tmp94 = tmp89 * tmp93
    tmp95 = tmp85 + tmp94
    tmp96 = tl.load(in_ptr1 + (tmp84 + 3*tmp78 + 9*x0), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr1 + (tmp87 + 3*tmp78 + 9*x0), xmask, eviction_policy='evict_last')
    tmp98 = tmp97 - tmp96
    tmp99 = tmp98 * tmp93
    tmp100 = tmp96 + tmp99
    tmp101 = tmp95 - tmp100
    tmp102 = tmp78.to(tl.float32)
    tmp103 = tmp77 - tmp102
    tmp104 = triton_helpers.maximum(tmp103, tmp8)
    tmp105 = triton_helpers.minimum(tmp104, tmp69)
    tmp106 = tmp101 * tmp105
    tmp107 = tmp100 + tmp106
    tmp108 = 1.6666666666666667
    tmp109 = tmp16 * tmp108
    tmp110 = triton_helpers.maximum(tmp109, tmp8)
    tmp111 = tmp110.to(tl.int32)
    tmp112 = tmp111 + tmp21
    tmp113 = tl.full([1, 1], 5, tl.int64)
    tmp114 = triton_helpers.minimum(tmp112, tmp113)
    tmp115 = tmp25 * tmp108
    tmp116 = triton_helpers.maximum(tmp115, tmp8)
    tmp117 = tmp116.to(tl.int32)
    tmp118 = tl.load(in_ptr2 + (tmp117 + 6*tmp114 + 36*x0), xmask, eviction_policy='evict_last')
    tmp119 = tmp117 + tmp21
    tmp120 = triton_helpers.minimum(tmp119, tmp113)
    tmp121 = tl.load(in_ptr2 + (tmp120 + 6*tmp114 + 36*x0), xmask, eviction_policy='evict_last')
    tmp122 = tmp121 - tmp118
    tmp123 = tmp117.to(tl.float32)
    tmp124 = tmp116 - tmp123
    tmp125 = triton_helpers.maximum(tmp124, tmp8)
    tmp126 = triton_helpers.minimum(tmp125, tmp69)
    tmp127 = tmp122 * tmp126
    tmp128 = tmp118 + tmp127
    tmp129 = tl.load(in_ptr2 + (tmp117 + 6*tmp111 + 36*x0), xmask, eviction_policy='evict_last')
    tmp130 = tl.load(in_ptr2 + (tmp120 + 6*tmp111 + 36*x0), xmask, eviction_policy='evict_last')
    tmp131 = tmp130 - tmp129
    tmp132 = tmp131 * tmp126
    tmp133 = tmp129 + tmp132
    tmp134 = tmp128 - tmp133
    tmp135 = tmp111.to(tl.float32)
    tmp136 = tmp110 - tmp135
    tmp137 = triton_helpers.maximum(tmp136, tmp8)
    tmp138 = triton_helpers.minimum(tmp137, tmp69)
    tmp139 = tmp134 * tmp138
    tmp140 = tmp133 + tmp139
    tmp141 = tmp72 - tmp74
    tmp142 = tmp20.to(tl.float32)
    tmp143 = tmp19 - tmp142
    tmp144 = triton_helpers.maximum(tmp143, tmp8)
    tmp145 = triton_helpers.minimum(tmp144, tmp69)
    tmp146 = tmp141 * tmp145
    tmp147 = tmp74 + tmp146
    tmp148 = tmp14 + tmp147
    tmp149 = tmp148 + tmp107
    tmp150 = tmp149 + tmp140
    tl.store(in_out_ptr4 + (r1 + 16*x0), tmp150, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf6 = empty_strided_cuda((4, 4, 3, 3), (36, 9, 3, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten._adaptive_avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_0.run(arg0_1, buf6, 144, grid=grid(144), stream=stream0)
        buf9 = empty_strided_cuda((4, 4, 6, 6), (144, 36, 6, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten._adaptive_avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d_1.run(arg0_1, buf9, 576, grid=grid(576), stream=stream0)
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        buf12 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x, x_1, feat, x_2, x_3, feat_1, x_5, feat_2, x_7, feat_3], Original ATen: [aten.mean, aten._to_copy, aten.arange, aten.mul, aten.clamp, aten._unsafe_index, aten.sub, aten.add, aten._adaptive_avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_per_fused__adaptive_avg_pool2d__to_copy__unsafe_index_add_arange_clamp_mean_mul_sub_2.run(buf12, arg0_1, buf6, buf9, 16, 16, grid=grid(16), stream=stream0)
        del arg0_1
        del buf6
        del buf9
    return (buf12, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
