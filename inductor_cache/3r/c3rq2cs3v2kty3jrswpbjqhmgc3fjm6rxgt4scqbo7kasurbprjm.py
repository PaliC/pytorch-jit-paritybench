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


# kernel path: inductor_cache/2u/c2ude6uxibqwbglowlhzlo3fyvxykwgakgeq43vshnrm2zk4gvk2.py
# Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   patch_pos_embed_1 => add_2, clamp_max_18, clamp_min_18, convert_element_type_3, floor_1
# Graph fragment:
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze,), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 1), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_2, 0), kwargs = {})
#   %clamp_max_18 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 13), kwargs = {})
triton_poi_fused__to_copy_add_clamp_floor_0 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_floor_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_floor_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_floor_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 3.414634146341464
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 13, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/th/cthr5ouniwor6t43nwbce4irxrvec7yns3zch2tkinxsk2csbsse.py
# Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
# Source node to ATen node mapping:
#   patch_pos_embed_1 => add_3, clamp_max_26, clamp_min_26, convert_element_type_3, floor_1
# Graph fragment:
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze,), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_3, 2), kwargs = {})
#   %clamp_min_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0), kwargs = {})
#   %clamp_max_26 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_26, 13), kwargs = {})
triton_poi_fused__to_copy_add_clamp_floor_1 = async_compile.triton('triton_poi_fused__to_copy_add_clamp_floor_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_clamp_floor_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_clamp_floor_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 3.414634146341464
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 13, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pr/cpri344roqxsmh6op6fz3a3l5ui4vjlrb7n2yxrzy3sdkebkn2az.py
# Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   patch_pos_embed_1 => clamp_max_2, clamp_min_2, convert_element_type_3, floor_1, sub_4
# Graph fragment:
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze,), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, 1), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_4, 0), kwargs = {})
#   %clamp_max_2 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 13), kwargs = {})
triton_poi_fused__to_copy_clamp_floor_sub_2 = async_compile.triton('triton_poi_fused__to_copy_clamp_floor_sub_2', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_floor_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_floor_sub_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 3.414634146341464
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = triton_helpers.maximum(tmp10, tmp11)
    tmp13 = tl.full([1], 13, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5l/c5ltws5k27j7htoorqxktmhupcdpedesmsapzffk3ysyrzbkncxn.py
# Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   patch_pos_embed_1 => add, clamp_max_5, clamp_min_5, convert_element_type, convert_element_type_2, floor, iota, mul, sub
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 3.414634146341464), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%sub,), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 0), kwargs = {})
#   %clamp_max_5 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 13), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 3.414634146341464
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 - tmp2
    tmp7 = libdevice.floor(tmp6)
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = triton_helpers.maximum(tmp8, tmp9)
    tmp11 = tl.full([1], 13, tl.int64)
    tmp12 = triton_helpers.minimum(tmp10, tmp11)
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ol/colbytc3gfhqkbojcnlx4xgbielcxhvaswagaoryvq7l2yq52rgk.py
# Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp, aten.rsub, aten._unsafe_index]
# Source node to ATen node mapping:
#   patch_pos_embed_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add, add_10, add_11, add_12, add_13, add_14, add_15, add_16, add_17, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_30, add_6, add_7, add_8, add_9, clamp_max, clamp_max_1, clamp_min, clamp_min_1, convert_element_type, floor, floor_1, iota, mul, mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_3, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_4, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_5, mul_6, mul_7, mul_8, mul_9, sub, sub_10, sub_11, sub_12, sub_13, sub_14, sub_15, sub_16, sub_17, sub_18, sub_19, sub_2, sub_20, sub_21, sub_3, sub_6, sub_7, sub_8, sub_9
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 3.414634146341464), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%sub,), kwargs = {})
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze, %floor_1), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_2, 0.0), kwargs = {})
#   %clamp_max : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 1.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %floor), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_3, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=6] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 1.0), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max_1, 1.0), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, -0.75), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, -3.75), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %add_6), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, -6.0), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_7, %add_6), kwargs = {})
#   %sub_7 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_4, -3.0), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_1, 1.25), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_5, 2.25), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %clamp_max_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %clamp_max_1), kwargs = {})
#   %add_8 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, 1), kwargs = {})
#   %sub_9 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, 1.25), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_8, 2.25), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %sub_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_9, %sub_9), kwargs = {})
#   %add_9 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, 1), kwargs = {})
#   %sub_11 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max_1), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, -0.75), kwargs = {})
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_11, -3.75), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %sub_11), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, -6.0), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_10, %sub_11), kwargs = {})
#   %sub_13 : [num_users=4] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_13, -3.0), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%clamp_max, 1.0), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, -0.75), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_14, -3.75), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %add_11), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, -6.0), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_12, %add_11), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_16, -3.0), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, 1.25), kwargs = {})
#   %sub_16 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_17, 2.25), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_16, %clamp_max), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_18, %clamp_max), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_19, 1), kwargs = {})
#   %sub_17 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %clamp_max), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, 1.25), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_20, 2.25), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %sub_17), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_21, %sub_17), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_22, 1), kwargs = {})
#   %sub_19 : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (2.0, %clamp_max), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, -0.75), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_23, -3.75), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %sub_19), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_24, -6.0), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_15, %sub_19), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_25, -3.0), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_2, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_2, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_2, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_2, %clamp_max_9]), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, %sub_7), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_1, %add_8), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_2, %add_9), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %mul_28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_3, %sub_13), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %mul_29), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_10, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_10, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_10, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_10, %clamp_max_9]), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_4, %sub_7), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_5, %add_8), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %mul_31), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_6, %add_9), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %mul_32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_7, %sub_13), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %mul_33), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_18, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_18, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_18, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_18, %clamp_max_9]), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_8, %sub_7), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_9, %add_8), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %mul_35), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_10, %add_9), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %mul_36), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_11, %sub_13), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %mul_37), kwargs = {})
#   %_unsafe_index_12 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_26, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_26, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_14 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_26, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max_26, %clamp_max_9]), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_12, %sub_7), kwargs = {})
#   %mul_39 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_13, %add_8), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %mul_39), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_14, %add_9), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_25, %mul_40), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_15, %sub_13), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %mul_41), kwargs = {})
#   %mul_42 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_18, %sub_15), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_21, %add_13), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_42, %mul_43), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_24, %add_14), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %mul_44), kwargs = {})
#   %mul_45 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_27, %sub_21), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %mul_45), kwargs = {})
triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4 = async_compile.triton('triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp77 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp92 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 14, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (768 + x2 + 768*tmp8 + 10752*tmp4), None, eviction_policy='evict_last')
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.5
    tmp13 = tmp11 + tmp12
    tmp14 = 3.414634146341464
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 - tmp12
    tmp17 = libdevice.floor(tmp16)
    tmp18 = tmp16 - tmp17
    tmp19 = 0.0
    tmp20 = triton_helpers.maximum(tmp18, tmp19)
    tmp21 = 1.0
    tmp22 = triton_helpers.minimum(tmp20, tmp21)
    tmp23 = tmp22 + tmp21
    tmp24 = -0.75
    tmp25 = tmp23 * tmp24
    tmp26 = -3.75
    tmp27 = tmp25 - tmp26
    tmp28 = tmp27 * tmp23
    tmp29 = -6.0
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30 * tmp23
    tmp32 = -3.0
    tmp33 = tmp31 - tmp32
    tmp34 = tmp9 * tmp33
    tmp36 = tmp35 + tmp1
    tmp37 = tmp35 < 0
    tmp38 = tl.where(tmp37, tmp36, tmp35)
    tmp39 = tl.load(in_ptr2 + (768 + x2 + 768*tmp38 + 10752*tmp4), None, eviction_policy='evict_last')
    tmp40 = 1.25
    tmp41 = tmp22 * tmp40
    tmp42 = 2.25
    tmp43 = tmp41 - tmp42
    tmp44 = tmp43 * tmp22
    tmp45 = tmp44 * tmp22
    tmp46 = tmp45 + tmp21
    tmp47 = tmp39 * tmp46
    tmp48 = tmp34 + tmp47
    tmp50 = tmp49 + tmp1
    tmp51 = tmp49 < 0
    tmp52 = tl.where(tmp51, tmp50, tmp49)
    tmp53 = tl.load(in_ptr2 + (768 + x2 + 768*tmp52 + 10752*tmp4), None, eviction_policy='evict_last')
    tmp54 = tmp21 - tmp22
    tmp55 = tmp54 * tmp40
    tmp56 = tmp55 - tmp42
    tmp57 = tmp56 * tmp54
    tmp58 = tmp57 * tmp54
    tmp59 = tmp58 + tmp21
    tmp60 = tmp53 * tmp59
    tmp61 = tmp48 + tmp60
    tmp63 = tmp62 + tmp1
    tmp64 = tmp62 < 0
    tmp65 = tl.where(tmp64, tmp63, tmp62)
    tmp66 = tl.load(in_ptr2 + (768 + x2 + 768*tmp65 + 10752*tmp4), None, eviction_policy='evict_last')
    tmp67 = 2.0
    tmp68 = tmp67 - tmp22
    tmp69 = tmp68 * tmp24
    tmp70 = tmp69 - tmp26
    tmp71 = tmp70 * tmp68
    tmp72 = tmp71 + tmp29
    tmp73 = tmp72 * tmp68
    tmp74 = tmp73 - tmp32
    tmp75 = tmp66 * tmp74
    tmp76 = tmp61 + tmp75
    tmp78 = tmp77 + tmp1
    tmp79 = tmp77 < 0
    tmp80 = tl.where(tmp79, tmp78, tmp77)
    tmp81 = tl.load(in_ptr2 + (768 + x2 + 768*tmp8 + 10752*tmp80), None, eviction_policy='evict_last')
    tmp82 = tmp81 * tmp33
    tmp83 = tl.load(in_ptr2 + (768 + x2 + 768*tmp38 + 10752*tmp80), None, eviction_policy='evict_last')
    tmp84 = tmp83 * tmp46
    tmp85 = tmp82 + tmp84
    tmp86 = tl.load(in_ptr2 + (768 + x2 + 768*tmp52 + 10752*tmp80), None, eviction_policy='evict_last')
    tmp87 = tmp86 * tmp59
    tmp88 = tmp85 + tmp87
    tmp89 = tl.load(in_ptr2 + (768 + x2 + 768*tmp65 + 10752*tmp80), None, eviction_policy='evict_last')
    tmp90 = tmp89 * tmp74
    tmp91 = tmp88 + tmp90
    tmp93 = tmp92 + tmp1
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tmp96 = tl.load(in_ptr2 + (768 + x2 + 768*tmp8 + 10752*tmp95), None, eviction_policy='evict_last')
    tmp97 = tmp96 * tmp33
    tmp98 = tl.load(in_ptr2 + (768 + x2 + 768*tmp38 + 10752*tmp95), None, eviction_policy='evict_last')
    tmp99 = tmp98 * tmp46
    tmp100 = tmp97 + tmp99
    tmp101 = tl.load(in_ptr2 + (768 + x2 + 768*tmp52 + 10752*tmp95), None, eviction_policy='evict_last')
    tmp102 = tmp101 * tmp59
    tmp103 = tmp100 + tmp102
    tmp104 = tl.load(in_ptr2 + (768 + x2 + 768*tmp65 + 10752*tmp95), None, eviction_policy='evict_last')
    tmp105 = tmp104 * tmp74
    tmp106 = tmp103 + tmp105
    tmp108 = tmp107 + tmp1
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tl.load(in_ptr2 + (768 + x2 + 768*tmp8 + 10752*tmp110), None, eviction_policy='evict_last')
    tmp112 = tmp111 * tmp33
    tmp113 = tl.load(in_ptr2 + (768 + x2 + 768*tmp38 + 10752*tmp110), None, eviction_policy='evict_last')
    tmp114 = tmp113 * tmp46
    tmp115 = tmp112 + tmp114
    tmp116 = tl.load(in_ptr2 + (768 + x2 + 768*tmp52 + 10752*tmp110), None, eviction_policy='evict_last')
    tmp117 = tmp116 * tmp59
    tmp118 = tmp115 + tmp117
    tmp119 = tl.load(in_ptr2 + (768 + x2 + 768*tmp65 + 10752*tmp110), None, eviction_policy='evict_last')
    tmp120 = tmp119 * tmp74
    tmp121 = tmp118 + tmp120
    tmp122 = x1
    tmp123 = tmp122.to(tl.float32)
    tmp124 = tmp123 + tmp12
    tmp125 = tmp124 * tmp14
    tmp126 = tmp125 - tmp12
    tmp127 = libdevice.floor(tmp126)
    tmp128 = tmp126 - tmp127
    tmp129 = triton_helpers.maximum(tmp128, tmp19)
    tmp130 = triton_helpers.minimum(tmp129, tmp21)
    tmp131 = tmp130 + tmp21
    tmp132 = tmp131 * tmp24
    tmp133 = tmp132 - tmp26
    tmp134 = tmp133 * tmp131
    tmp135 = tmp134 + tmp29
    tmp136 = tmp135 * tmp131
    tmp137 = tmp136 - tmp32
    tmp138 = tmp76 * tmp137
    tmp139 = tmp130 * tmp40
    tmp140 = tmp139 - tmp42
    tmp141 = tmp140 * tmp130
    tmp142 = tmp141 * tmp130
    tmp143 = tmp142 + tmp21
    tmp144 = tmp91 * tmp143
    tmp145 = tmp138 + tmp144
    tmp146 = tmp21 - tmp130
    tmp147 = tmp146 * tmp40
    tmp148 = tmp147 - tmp42
    tmp149 = tmp148 * tmp146
    tmp150 = tmp149 * tmp146
    tmp151 = tmp150 + tmp21
    tmp152 = tmp106 * tmp151
    tmp153 = tmp145 + tmp152
    tmp154 = tmp67 - tmp130
    tmp155 = tmp154 * tmp24
    tmp156 = tmp155 - tmp26
    tmp157 = tmp156 * tmp154
    tmp158 = tmp157 + tmp29
    tmp159 = tmp158 * tmp154
    tmp160 = tmp159 - tmp32
    tmp161 = tmp121 * tmp160
    tmp162 = tmp153 + tmp161
    tl.store(in_out_ptr0 + (x4), tmp162, None)
''', device_str='cuda')


# kernel path: inductor_cache/z5/cz5kowsyn6petsizholzymbqv7kus5umn2mrgxiigqs5pvmqqdx6.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_1, %view_2], 1), kwargs = {})
triton_poi_fused_cat_5 = async_compile.triton('triton_poi_fused_cat_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 768
    x0 = (xindex % 768)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 17, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (16*x0 + ((((-1) + x1) % 16))), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u4/cu4lf52tjilpns5v26o6pj24w37gxw2m7zkwrvpjaim2yrhgjew6.py
# Topologically Sorted Source Nodes: [x_1, x_2, layer_norm], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add_32, add_33, mul_46, mul_47, rsqrt, sub_22, var_mean
#   x_1 => cat
#   x_2 => add_31
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%expand, %permute], 1), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %cat_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_31, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_32,), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_31, %getitem_1), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %primals_6), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %primals_7), kwargs = {})
triton_red_fused_add_cat_native_layer_norm_6 = async_compile.triton('triton_red_fused_add_cat_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_native_layer_norm_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_native_layer_norm_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 68
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 17)
    x1 = xindex // 17
    x3 = xindex
    tmp18_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp18_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr3 + (r2 + 768*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = tl.full([1, 1], 1, tl.int64)
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp0 >= tmp3
        tmp7 = tl.full([1, 1], 17, tl.int64)
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr1 + (16*r2 + 12288*x1 + ((((-1) + x0) % 16))), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr2 + (tl.broadcast_to(r2, [XBLOCK, RBLOCK])), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp11 = tmp9 + tmp10
        tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
        tmp13 = tl.where(tmp6, tmp11, tmp12)
        tmp14 = tl.where(tmp4, tmp5, tmp13)
        tmp16 = tmp14 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp18_mean_next, tmp18_m2_next, tmp18_weight_next = triton_helpers.welford_reduce(
            tmp17, tmp18_mean, tmp18_m2, tmp18_weight, roffset == 0
        )
        tmp18_mean = tl.where(rmask & xmask, tmp18_mean_next, tmp18_mean)
        tmp18_m2 = tl.where(rmask & xmask, tmp18_m2_next, tmp18_m2)
        tmp18_weight = tl.where(rmask & xmask, tmp18_weight_next, tmp18_weight)
        tl.store(out_ptr0 + (r2 + 768*x3), tmp14, rmask & xmask)
    tmp18_tmp, tmp19_tmp, tmp20_tmp = triton_helpers.welford(
        tmp18_mean, tmp18_m2, tmp18_weight, 1
    )
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tl.store(out_ptr1 + (x3), tmp18, xmask)
    tmp21 = 768.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp25, xmask)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp26 = tl.load(out_ptr0 + (r2 + 768*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp27 = tl.load(in_ptr3 + (r2 + 768*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp31 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp33 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp28 - tmp18
        tmp30 = tmp29 * tmp25
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr2 + (r2 + 768*x3), tmp34, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qr/cqrxzi75dghb2uaqw6w3ckbuc2vu5duq2duz2zv3xnfboy5haiqy.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 17)
    x2 = ((xindex // 1088) % 12)
    x3 = xindex // 13056
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 2304*x1 + 39168*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pt/cptmvrm5j5nmdnvcuepae67pi2xlcimxbpikgnqupo7mdn5ah44e.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4096, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 17
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 768)
    y1 = yindex // 768
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (768 + y0 + 2304*x2 + 39168*y1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 17*y3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lf/clffunojwetdne43dccnw7oezn4hxgmc23e52s7co43dasizgtrc.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div, exp, sum_1
# Graph fragment:
#   %mul_tensor_22 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_8, 1), kwargs = {})
#   %amax_default_11 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_22, [-1], True), kwargs = {})
#   %sub_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_22, %amax_default_11), kwargs = {})
#   %mul_tensor_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_11, 0.125), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_23,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_per_fused__softmax_9 = async_compile.triton('triton_per_fused__softmax_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__softmax_9(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 816
    rnumel = 17
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    x2 = (xindex % 204)
    x3 = xindex // 204
    tmp0 = tl.load(in_ptr0 + (r1 + 17*x0), rmask & xmask, other=0.0)
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None]
    tmp7 = tmp2 - tmp6
    tmp8 = 0.125
    tmp9 = tmp7 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp13 = tl.where(rmask & xmask, tmp11, 0)
    tmp14 = tl.sum(tmp13, 1)[:, None]
    tmp15 = tmp10 / tmp14
    tl.store(out_ptr2 + (r1 + 17*x2 + 3488*x3), tmp15, rmask & xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ib/cibwdyll4kjapyibgpcsceaitginaybtyqrugziqchelns7rixmv.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   matmul_1 => bmm_1
# Graph fragment:
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_9, %view_10), kwargs = {})
triton_poi_fused_bmm_10 = async_compile.triton('triton_poi_fused_bmm_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 289)
    x1 = xindex // 289
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 289*((x1 % 12)) + 3488*(x1 // 12)), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/n5/cn5xrsxzj5qugbqwsfaxzknoak2hbht5uynpquqqxheph2g3tqvg.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_4,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_11 = async_compile.triton('triton_poi_fused_clone_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 17)
    x2 = ((xindex // 1088) % 12)
    x3 = xindex // 13056
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (1536 + x0 + 64*x2 + 2304*x1 + 39168*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vt/cvtxvdhuit62sgspx2rhkedgv26cgy7jare4jggalz4ekwymqj5v.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_4 => clone_6
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_12 = async_compile.triton('triton_poi_fused_clone_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 12)
    x2 = ((xindex // 768) % 17)
    x3 = xindex // 13056
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 1088*x1 + 13056*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gk/cgkyjmi2dki36atb72mm5qpj7rcwyhtfpgzzmbu2a73zlawbvh7l.py
# Topologically Sorted Source Nodes: [x_2, x_7, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_1 => add_35, add_36, mul_49, mul_50, rsqrt_1, sub_24, var_mean_1
#   x_2 => add_31
#   x_7 => add_34
# Graph fragment:
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %cat_1), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %view_14), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_34, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_35,), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %getitem_3), kwargs = {})
#   %mul_49 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_1), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %primals_11), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %primals_12), kwargs = {})
#   %div_38 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 68
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 17)
    tmp0 = tl.load(in_ptr0 + (r2 + 768*x3), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + 768*x0), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r2 + 768*x3), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = 0.0013020833333333333
    tmp35 = tmp28 * tmp34
    tl.store(out_ptr2 + (r2 + 768*x3), tmp29, rmask)
    tl.store(out_ptr3 + (r2 + 768*x3), tmp33, rmask)
    tl.store(out_ptr4 + (x3), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/ur/curx5pq4djt2oaznp4eoenruyrenbo6jaws5alh2fe26p4o3fv7u.py
# Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_9 => add_37, erf, mul_51, mul_52, mul_53
# Graph fragment:
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.5), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_16, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_52,), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_51, %add_37), kwargs = {})
triton_poi_fused_gelu_14 = async_compile.triton('triton_poi_fused_gelu_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 208896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/gf/cgfyp7tlhgfvm52u2ai2ij4i5jrrocvtnq7hl7rfcslbmtr5c75t.py
# Topologically Sorted Source Nodes: [x_2, x_7, x_13, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_2 => add_39, add_40, mul_54, mul_55, rsqrt_2, sub_25, var_mean_2
#   x_13 => add_38
#   x_2 => add_31
#   x_7 => add_34
# Graph fragment:
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat, %cat_1), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_31, %view_14), kwargs = {})
#   %add_38 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %view_18), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_38, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_39 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_39,), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_38, %getitem_5), kwargs = {})
#   %mul_54 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_2), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_54, %primals_17), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_55, %primals_18), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 68
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = (xindex % 17)
    tmp0 = tl.load(in_ptr0 + (r2 + 768*x3), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r2 + 768*x0), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r2 + 768*x3), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r2 + 768*x3), rmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = 0.0013020833333333333
    tmp39 = tmp32 * tmp38
    tl.store(in_out_ptr0 + (r2 + 768*x3), tmp10, rmask)
    tl.store(out_ptr2 + (r2 + 768*x3), tmp33, rmask)
    tl.store(out_ptr3 + (r2 + 768*x3), tmp37, rmask)
    tl.store(out_ptr4 + (x3), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/gp/cgpcbe2sskjc4kb5b7zxspm47amfwuxfp3ahibp6llp5c22q7r3f.py
# Topologically Sorted Source Nodes: [x_17, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_3 => add_42, add_43, mul_57, mul_58, rsqrt_3, sub_27, var_mean_3
#   x_17 => add_41
# Graph fragment:
#   %add_41 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %view_30), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_41, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_42,), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_41, %getitem_7), kwargs = {})
#   %mul_57 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_3), kwargs = {})
#   %mul_58 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_57, %primals_22), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_58, %primals_23), kwargs = {})
#   %div_36 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 68
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 768*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 768.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0013020833333333333
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r1 + 768*x0), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + 768*x0), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/7j/c7joofmkfeka44oa7vvkt5phcyks5z75enzz7n6ai4zaqxcexd7v.py
# Topologically Sorted Source Nodes: [x_17, x_23, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_4 => add_46, add_47, mul_62, mul_63, rsqrt_4, sub_28, var_mean_4
#   x_17 => add_41
#   x_23 => add_45
# Graph fragment:
#   %add_41 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_38, %view_30), kwargs = {})
#   %add_45 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_41, %view_34), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_45, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_46,), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_45, %getitem_9), kwargs = {})
#   %mul_62 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_4), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_62, %primals_28), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %primals_29), kwargs = {})
#   %div_35 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 68
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 768*x0), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0013020833333333333
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + 768*x0), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + 768*x0), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/5s/c5sekr4354rtzeq4maekwr5oympyemklqhm7w2x7rmktvzeljn7a.py
# Topologically Sorted Source Nodes: [x_117, x_123, x_127], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_117 => add_117
#   x_123 => add_121
#   x_127 => add_122, mul_148, rsqrt_27, sub_61, var_mean_27
# Graph fragment:
#   %add_117 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_112, %view_190), kwargs = {})
#   %add_121 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_117, %view_194), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_121, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_122 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_54, 1e-05), kwargs = {})
#   %rsqrt_27 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_122,), kwargs = {})
#   %sub_61 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_121, %getitem_55), kwargs = {})
#   %mul_148 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_61, %rsqrt_27), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_27, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 128, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 68
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 768*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 768*x0), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 768, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 768.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = 0.0013020833333333333
    tmp33 = tmp30 * tmp32
    tl.store(in_out_ptr0 + (r1 + 768*x0), tmp31, rmask)
    tl.store(out_ptr2 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/vs/cvsizdsv5kgpb6367xx3d7tmtkuhtodbv26bul7l4ma54yxpvpy6.py
# Topologically Sorted Source Nodes: [embed], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   embed => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%select_37, %select_38, %select_39, %select_40], -1), kwargs = {})
triton_poi_fused_cat_19 = async_compile.triton('triton_poi_fused_cat_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 3072)
    x1 = xindex // 3072
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (13056*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = tl.load(in_ptr2 + (x0), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1536, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tl.load(in_ptr3 + (13056*x1 + ((-768) + x0)), tmp15, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.load(in_ptr1 + ((-768) + x0), tmp15, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr2 + ((-768) + x0), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp15, tmp20, tmp21)
    tmp23 = tmp0 >= tmp13
    tmp24 = tl.full([1], 2304, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr4 + (13056*x1 + ((-1536) + x0)), tmp26, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr1 + ((-1536) + x0), tmp26, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 * tmp28
    tmp30 = tl.load(in_ptr2 + ((-1536) + x0), tmp26, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp26, tmp31, tmp32)
    tmp34 = tmp0 >= tmp24
    tmp35 = tl.full([1], 3072, tl.int64)
    tmp36 = tmp0 < tmp35
    tmp37 = tl.load(in_ptr5 + (13056*x1 + ((-2304) + x0)), tmp34, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.load(in_ptr1 + ((-2304) + x0), tmp34, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 * tmp38
    tmp40 = tl.load(in_ptr2 + ((-2304) + x0), tmp34, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 + tmp40
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp34, tmp41, tmp42)
    tmp44 = tl.where(tmp26, tmp33, tmp43)
    tmp45 = tl.where(tmp15, tmp22, tmp44)
    tmp46 = tl.where(tmp4, tmp11, tmp45)
    tl.store(out_ptr0 + (x2), tmp46, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139 = args
    args.clear()
    assert_size_stride(primals_1, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_2, (768, 3, 16, 16), (768, 256, 16, 1))
    assert_size_stride(primals_3, (768, ), (1, ))
    assert_size_stride(primals_4, (1, 1, 768), (768, 768, 1))
    assert_size_stride(primals_5, (1, 197, 768), (151296, 768, 1))
    assert_size_stride(primals_6, (768, ), (1, ))
    assert_size_stride(primals_7, (768, ), (1, ))
    assert_size_stride(primals_8, (2304, 768), (768, 1))
    assert_size_stride(primals_9, (768, 768), (768, 1))
    assert_size_stride(primals_10, (768, ), (1, ))
    assert_size_stride(primals_11, (768, ), (1, ))
    assert_size_stride(primals_12, (768, ), (1, ))
    assert_size_stride(primals_13, (3072, 768), (768, 1))
    assert_size_stride(primals_14, (3072, ), (1, ))
    assert_size_stride(primals_15, (768, 3072), (3072, 1))
    assert_size_stride(primals_16, (768, ), (1, ))
    assert_size_stride(primals_17, (768, ), (1, ))
    assert_size_stride(primals_18, (768, ), (1, ))
    assert_size_stride(primals_19, (2304, 768), (768, 1))
    assert_size_stride(primals_20, (768, 768), (768, 1))
    assert_size_stride(primals_21, (768, ), (1, ))
    assert_size_stride(primals_22, (768, ), (1, ))
    assert_size_stride(primals_23, (768, ), (1, ))
    assert_size_stride(primals_24, (3072, 768), (768, 1))
    assert_size_stride(primals_25, (3072, ), (1, ))
    assert_size_stride(primals_26, (768, 3072), (3072, 1))
    assert_size_stride(primals_27, (768, ), (1, ))
    assert_size_stride(primals_28, (768, ), (1, ))
    assert_size_stride(primals_29, (768, ), (1, ))
    assert_size_stride(primals_30, (2304, 768), (768, 1))
    assert_size_stride(primals_31, (768, 768), (768, 1))
    assert_size_stride(primals_32, (768, ), (1, ))
    assert_size_stride(primals_33, (768, ), (1, ))
    assert_size_stride(primals_34, (768, ), (1, ))
    assert_size_stride(primals_35, (3072, 768), (768, 1))
    assert_size_stride(primals_36, (3072, ), (1, ))
    assert_size_stride(primals_37, (768, 3072), (3072, 1))
    assert_size_stride(primals_38, (768, ), (1, ))
    assert_size_stride(primals_39, (768, ), (1, ))
    assert_size_stride(primals_40, (768, ), (1, ))
    assert_size_stride(primals_41, (2304, 768), (768, 1))
    assert_size_stride(primals_42, (768, 768), (768, 1))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (768, ), (1, ))
    assert_size_stride(primals_46, (3072, 768), (768, 1))
    assert_size_stride(primals_47, (3072, ), (1, ))
    assert_size_stride(primals_48, (768, 3072), (3072, 1))
    assert_size_stride(primals_49, (768, ), (1, ))
    assert_size_stride(primals_50, (768, ), (1, ))
    assert_size_stride(primals_51, (768, ), (1, ))
    assert_size_stride(primals_52, (2304, 768), (768, 1))
    assert_size_stride(primals_53, (768, 768), (768, 1))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (768, ), (1, ))
    assert_size_stride(primals_56, (768, ), (1, ))
    assert_size_stride(primals_57, (3072, 768), (768, 1))
    assert_size_stride(primals_58, (3072, ), (1, ))
    assert_size_stride(primals_59, (768, 3072), (3072, 1))
    assert_size_stride(primals_60, (768, ), (1, ))
    assert_size_stride(primals_61, (768, ), (1, ))
    assert_size_stride(primals_62, (768, ), (1, ))
    assert_size_stride(primals_63, (2304, 768), (768, 1))
    assert_size_stride(primals_64, (768, 768), (768, 1))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, ), (1, ))
    assert_size_stride(primals_68, (3072, 768), (768, 1))
    assert_size_stride(primals_69, (3072, ), (1, ))
    assert_size_stride(primals_70, (768, 3072), (3072, 1))
    assert_size_stride(primals_71, (768, ), (1, ))
    assert_size_stride(primals_72, (768, ), (1, ))
    assert_size_stride(primals_73, (768, ), (1, ))
    assert_size_stride(primals_74, (2304, 768), (768, 1))
    assert_size_stride(primals_75, (768, 768), (768, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, ), (1, ))
    assert_size_stride(primals_78, (768, ), (1, ))
    assert_size_stride(primals_79, (3072, 768), (768, 1))
    assert_size_stride(primals_80, (3072, ), (1, ))
    assert_size_stride(primals_81, (768, 3072), (3072, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (768, ), (1, ))
    assert_size_stride(primals_84, (768, ), (1, ))
    assert_size_stride(primals_85, (2304, 768), (768, 1))
    assert_size_stride(primals_86, (768, 768), (768, 1))
    assert_size_stride(primals_87, (768, ), (1, ))
    assert_size_stride(primals_88, (768, ), (1, ))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (3072, 768), (768, 1))
    assert_size_stride(primals_91, (3072, ), (1, ))
    assert_size_stride(primals_92, (768, 3072), (3072, 1))
    assert_size_stride(primals_93, (768, ), (1, ))
    assert_size_stride(primals_94, (768, ), (1, ))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (2304, 768), (768, 1))
    assert_size_stride(primals_97, (768, 768), (768, 1))
    assert_size_stride(primals_98, (768, ), (1, ))
    assert_size_stride(primals_99, (768, ), (1, ))
    assert_size_stride(primals_100, (768, ), (1, ))
    assert_size_stride(primals_101, (3072, 768), (768, 1))
    assert_size_stride(primals_102, (3072, ), (1, ))
    assert_size_stride(primals_103, (768, 3072), (3072, 1))
    assert_size_stride(primals_104, (768, ), (1, ))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, ), (1, ))
    assert_size_stride(primals_107, (768, ), (1, ))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (2304, 768), (768, 1))
    assert_size_stride(primals_110, (768, 768), (768, 1))
    assert_size_stride(primals_111, (768, ), (1, ))
    assert_size_stride(primals_112, (768, ), (1, ))
    assert_size_stride(primals_113, (768, ), (1, ))
    assert_size_stride(primals_114, (3072, 768), (768, 1))
    assert_size_stride(primals_115, (3072, ), (1, ))
    assert_size_stride(primals_116, (768, 3072), (3072, 1))
    assert_size_stride(primals_117, (768, ), (1, ))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, ), (1, ))
    assert_size_stride(primals_120, (2304, 768), (768, 1))
    assert_size_stride(primals_121, (768, 768), (768, 1))
    assert_size_stride(primals_122, (768, ), (1, ))
    assert_size_stride(primals_123, (768, ), (1, ))
    assert_size_stride(primals_124, (768, ), (1, ))
    assert_size_stride(primals_125, (3072, 768), (768, 1))
    assert_size_stride(primals_126, (3072, ), (1, ))
    assert_size_stride(primals_127, (768, 3072), (3072, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, ), (1, ))
    assert_size_stride(primals_130, (768, ), (1, ))
    assert_size_stride(primals_131, (2304, 768), (768, 1))
    assert_size_stride(primals_132, (768, 768), (768, 1))
    assert_size_stride(primals_133, (768, ), (1, ))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (768, ), (1, ))
    assert_size_stride(primals_136, (3072, 768), (768, 1))
    assert_size_stride(primals_137, (3072, ), (1, ))
    assert_size_stride(primals_138, (768, 3072), (3072, 1))
    assert_size_stride(primals_139, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_1, primals_2, stride=(16, 16), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 768, 4, 4), (12288, 16, 4, 1))
        buf12 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clamp_floor_0.run(buf12, 4, grid=grid(4), stream=stream0)
        buf15 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clamp_floor_1.run(buf15, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_2.run(buf2, 4, grid=grid(4), stream=stream0)
        buf3 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_2.run(buf3, 4, grid=grid(4), stream=stream0)
        buf4 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3.run(buf4, 4, grid=grid(4), stream=stream0)
        buf5 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clamp_floor_0.run(buf5, 4, grid=grid(4), stream=stream0)
        buf6 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_clamp_floor_1.run(buf6, 4, grid=grid(4), stream=stream0)
        buf9 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.floor, aten._to_copy, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3.run(buf9, 4, grid=grid(4), stream=stream0)
        buf7 = empty_strided_cuda((1, 768, 4, 4), (12288, 16, 4, 1), torch.float32)
        buf8 = buf7; del buf7  # reuse
        buf18 = buf8; del buf8  # reuse
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [patch_pos_embed_1], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp, aten.rsub, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4.run(buf19, buf2, buf3, primals_5, buf4, buf5, buf6, buf9, buf12, buf15, 12288, grid=grid(12288), stream=stream0)
        buf20 = empty_strided_cuda((1, 17, 768), (13056, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_5.run(primals_5, buf19, buf20, 13056, grid=grid(13056), stream=stream0)
        del primals_5
        buf1 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf21 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        buf22 = empty_strided_cuda((4, 17, 1), (17, 1, 68), torch.float32)
        buf24 = reinterpret_tensor(buf22, (4, 17, 1), (17, 1, 1), 0); del buf22  # reuse
        buf25 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, layer_norm], Original ATen: [aten.cat, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_native_layer_norm_6.run(buf24, primals_4, buf0, primals_3, buf20, primals_6, primals_7, buf1, buf21, buf25, 68, 768, grid=grid(68), stream=stream0)
        del buf0
        del primals_3
        del primals_4
        del primals_7
        buf26 = empty_strided_cuda((68, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf25, (68, 768), (768, 1), 0), reinterpret_tensor(primals_8, (768, 2304), (1, 768), 0), out=buf26)
        buf27 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf26, buf27, 52224, grid=grid(52224), stream=stream0)
        buf28 = empty_strided_cuda((4, 12, 64, 17), (13056, 1088, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf26, buf28, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf29 = empty_strided_cuda((48, 17, 17), (289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf28, (48, 64, 17), (1088, 17, 1), 0), out=buf29)
        buf32 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf29, buf32, 816, 17, grid=grid(816), stream=stream0)
        buf33 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf32, buf33, 13872, grid=grid(13872), stream=stream0)
        buf34 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf26, buf34, 52224, grid=grid(52224), stream=stream0)
        buf35 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf33, reinterpret_tensor(buf34, (48, 17, 64), (1088, 64, 1), 0), out=buf35)
        buf36 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf35, buf36, 52224, grid=grid(52224), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (68, 768), (768, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf36, (68, 768), (768, 1), 0), reinterpret_tensor(primals_9, (768, 768), (1, 768), 0), out=buf37)
        buf41 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf42 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf361 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_7, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13.run(buf1, buf20, buf37, primals_10, primals_11, primals_12, buf41, buf42, buf361, 68, 768, grid=grid(68), stream=stream0)
        del primals_12
        buf43 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_14, reinterpret_tensor(buf42, (68, 768), (768, 1), 0), reinterpret_tensor(primals_13, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf43)
        del primals_14
        buf44 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf43, buf44, 208896, grid=grid(208896), stream=stream0)
        buf45 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf44, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_15, (3072, 768), (1, 3072), 0), out=buf45)
        buf46 = reinterpret_tensor(buf37, (4, 17, 768), (13056, 768, 1), 0); del buf37  # reuse
        buf50 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf51 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf360 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_7, x_13, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_15.run(buf46, buf1, buf20, primals_10, buf45, primals_16, primals_17, primals_18, buf50, buf51, buf360, 68, 768, grid=grid(68), stream=stream0)
        del primals_10
        del primals_16
        del primals_18
        buf52 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (68, 768), (768, 1), 0), reinterpret_tensor(primals_19, (768, 2304), (1, 768), 0), out=buf52)
        buf53 = reinterpret_tensor(buf45, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf45  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf52, buf53, 52224, grid=grid(52224), stream=stream0)
        buf54 = empty_strided_cuda((4, 12, 64, 17), (13056, 1088, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf52, buf54, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf55 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf54, (48, 64, 17), (1088, 17, 1), 0), out=buf55)
        buf58 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf55, buf58, 816, 17, grid=grid(816), stream=stream0)
        buf59 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf58, buf59, 13872, grid=grid(13872), stream=stream0)
        buf60 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf52, buf60, 52224, grid=grid(52224), stream=stream0)
        buf61 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf59, reinterpret_tensor(buf60, (48, 17, 64), (1088, 64, 1), 0), out=buf61)
        buf62 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf61, buf62, 52224, grid=grid(52224), stream=stream0)
        buf63 = reinterpret_tensor(buf61, (68, 768), (768, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf62, (68, 768), (768, 1), 0), reinterpret_tensor(primals_20, (768, 768), (1, 768), 0), out=buf63)
        buf67 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf68 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf359 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_17, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf46, buf63, primals_21, primals_22, primals_23, buf67, buf68, buf359, 68, 768, grid=grid(68), stream=stream0)
        del primals_23
        buf69 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_25, reinterpret_tensor(buf68, (68, 768), (768, 1), 0), reinterpret_tensor(primals_24, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf69)
        del primals_25
        buf70 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf69, buf70, 208896, grid=grid(208896), stream=stream0)
        buf71 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf70, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_26, (3072, 768), (1, 3072), 0), out=buf71)
        buf72 = buf46; del buf46  # reuse
        buf76 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf77 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf358 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_17, x_23, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf72, buf63, primals_21, buf71, primals_27, primals_28, primals_29, buf76, buf77, buf358, 68, 768, grid=grid(68), stream=stream0)
        del primals_21
        del primals_27
        del primals_29
        buf78 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (68, 768), (768, 1), 0), reinterpret_tensor(primals_30, (768, 2304), (1, 768), 0), out=buf78)
        buf79 = reinterpret_tensor(buf71, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf71  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf78, buf79, 52224, grid=grid(52224), stream=stream0)
        buf80 = reinterpret_tensor(buf63, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf78, buf80, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf81 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf79, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf80, (48, 64, 17), (1088, 17, 1), 0), out=buf81)
        buf84 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_7], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf81, buf84, 816, 17, grid=grid(816), stream=stream0)
        buf85 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf84, buf85, 13872, grid=grid(13872), stream=stream0)
        buf86 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf78, buf86, 52224, grid=grid(52224), stream=stream0)
        buf87 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf85, reinterpret_tensor(buf86, (48, 17, 64), (1088, 64, 1), 0), out=buf87)
        buf88 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf87, buf88, 52224, grid=grid(52224), stream=stream0)
        buf89 = reinterpret_tensor(buf87, (68, 768), (768, 1), 0); del buf87  # reuse
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf88, (68, 768), (768, 1), 0), reinterpret_tensor(primals_31, (768, 768), (1, 768), 0), out=buf89)
        buf93 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf94 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf357 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf72, buf89, primals_32, primals_33, primals_34, buf93, buf94, buf357, 68, 768, grid=grid(68), stream=stream0)
        del primals_34
        buf95 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_36, reinterpret_tensor(buf94, (68, 768), (768, 1), 0), reinterpret_tensor(primals_35, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf95)
        del primals_36
        buf96 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf95, buf96, 208896, grid=grid(208896), stream=stream0)
        buf97 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf96, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_37, (3072, 768), (1, 3072), 0), out=buf97)
        buf98 = buf72; del buf72  # reuse
        buf102 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf103 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf356 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_27, x_33, layer_norm_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf98, buf89, primals_32, buf97, primals_38, primals_39, primals_40, buf102, buf103, buf356, 68, 768, grid=grid(68), stream=stream0)
        del primals_32
        del primals_38
        del primals_40
        buf104 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf103, (68, 768), (768, 1), 0), reinterpret_tensor(primals_41, (768, 2304), (1, 768), 0), out=buf104)
        buf105 = reinterpret_tensor(buf97, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf104, buf105, 52224, grid=grid(52224), stream=stream0)
        buf106 = reinterpret_tensor(buf89, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf89  # reuse
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf104, buf106, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf107 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf106, (48, 64, 17), (1088, 17, 1), 0), out=buf107)
        buf110 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_10], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf107, buf110, 816, 17, grid=grid(816), stream=stream0)
        buf111 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf110, buf111, 13872, grid=grid(13872), stream=stream0)
        buf112 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf104, buf112, 52224, grid=grid(52224), stream=stream0)
        buf113 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf111, reinterpret_tensor(buf112, (48, 17, 64), (1088, 64, 1), 0), out=buf113)
        buf114 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf113, buf114, 52224, grid=grid(52224), stream=stream0)
        buf115 = reinterpret_tensor(buf113, (68, 768), (768, 1), 0); del buf113  # reuse
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf114, (68, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 768), (1, 768), 0), out=buf115)
        buf119 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf120 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf355 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_37, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf98, buf115, primals_43, primals_44, primals_45, buf119, buf120, buf355, 68, 768, grid=grid(68), stream=stream0)
        del primals_45
        buf121 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_47, reinterpret_tensor(buf120, (68, 768), (768, 1), 0), reinterpret_tensor(primals_46, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf121)
        del primals_47
        buf122 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf121, buf122, 208896, grid=grid(208896), stream=stream0)
        buf123 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf122, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_48, (3072, 768), (1, 3072), 0), out=buf123)
        buf124 = buf98; del buf98  # reuse
        buf128 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf129 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf354 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_37, x_43, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf124, buf115, primals_43, buf123, primals_49, primals_50, primals_51, buf128, buf129, buf354, 68, 768, grid=grid(68), stream=stream0)
        del primals_43
        del primals_49
        del primals_51
        buf130 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (68, 768), (768, 1), 0), reinterpret_tensor(primals_52, (768, 2304), (1, 768), 0), out=buf130)
        buf131 = reinterpret_tensor(buf123, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf130, buf131, 52224, grid=grid(52224), stream=stream0)
        buf132 = reinterpret_tensor(buf115, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf115  # reuse
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf130, buf132, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf133 = buf111; del buf111  # reuse
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf131, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf132, (48, 64, 17), (1088, 17, 1), 0), out=buf133)
        buf136 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_13], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf133, buf136, 816, 17, grid=grid(816), stream=stream0)
        buf137 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf136, buf137, 13872, grid=grid(13872), stream=stream0)
        buf138 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf130, buf138, 52224, grid=grid(52224), stream=stream0)
        buf139 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf137, reinterpret_tensor(buf138, (48, 17, 64), (1088, 64, 1), 0), out=buf139)
        buf140 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf139, buf140, 52224, grid=grid(52224), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (68, 768), (768, 1), 0); del buf139  # reuse
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf140, (68, 768), (768, 1), 0), reinterpret_tensor(primals_53, (768, 768), (1, 768), 0), out=buf141)
        buf145 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf146 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf353 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_47, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf124, buf141, primals_54, primals_55, primals_56, buf145, buf146, buf353, 68, 768, grid=grid(68), stream=stream0)
        del primals_56
        buf147 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_58, reinterpret_tensor(buf146, (68, 768), (768, 1), 0), reinterpret_tensor(primals_57, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf147)
        del primals_58
        buf148 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf147, buf148, 208896, grid=grid(208896), stream=stream0)
        buf149 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf148, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_59, (3072, 768), (1, 3072), 0), out=buf149)
        buf150 = buf124; del buf124  # reuse
        buf154 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf155 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf352 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_47, x_53, layer_norm_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf150, buf141, primals_54, buf149, primals_60, primals_61, primals_62, buf154, buf155, buf352, 68, 768, grid=grid(68), stream=stream0)
        del primals_54
        del primals_60
        del primals_62
        buf156 = buf130; del buf130  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf155, (68, 768), (768, 1), 0), reinterpret_tensor(primals_63, (768, 2304), (1, 768), 0), out=buf156)
        buf157 = reinterpret_tensor(buf149, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf149  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf156, buf157, 52224, grid=grid(52224), stream=stream0)
        buf158 = reinterpret_tensor(buf141, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf141  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf156, buf158, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf159 = buf137; del buf137  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf157, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf158, (48, 64, 17), (1088, 17, 1), 0), out=buf159)
        buf162 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_16], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf159, buf162, 816, 17, grid=grid(816), stream=stream0)
        buf163 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf162, buf163, 13872, grid=grid(13872), stream=stream0)
        buf164 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf156, buf164, 52224, grid=grid(52224), stream=stream0)
        buf165 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf163, reinterpret_tensor(buf164, (48, 17, 64), (1088, 64, 1), 0), out=buf165)
        buf166 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf165, buf166, 52224, grid=grid(52224), stream=stream0)
        buf167 = reinterpret_tensor(buf165, (68, 768), (768, 1), 0); del buf165  # reuse
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf166, (68, 768), (768, 1), 0), reinterpret_tensor(primals_64, (768, 768), (1, 768), 0), out=buf167)
        buf171 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf172 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf351 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_57, layer_norm_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf150, buf167, primals_65, primals_66, primals_67, buf171, buf172, buf351, 68, 768, grid=grid(68), stream=stream0)
        del primals_67
        buf173 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_69, reinterpret_tensor(buf172, (68, 768), (768, 1), 0), reinterpret_tensor(primals_68, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf173)
        del primals_69
        buf174 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf173, buf174, 208896, grid=grid(208896), stream=stream0)
        buf175 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_61], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf174, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_70, (3072, 768), (1, 3072), 0), out=buf175)
        buf176 = buf150; del buf150  # reuse
        buf180 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf181 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf350 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_57, x_63, layer_norm_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf176, buf167, primals_65, buf175, primals_71, primals_72, primals_73, buf180, buf181, buf350, 68, 768, grid=grid(68), stream=stream0)
        del primals_65
        del primals_71
        del primals_73
        buf182 = buf156; del buf156  # reuse
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (68, 768), (768, 1), 0), reinterpret_tensor(primals_74, (768, 2304), (1, 768), 0), out=buf182)
        buf183 = reinterpret_tensor(buf175, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf175  # reuse
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf182, buf183, 52224, grid=grid(52224), stream=stream0)
        buf184 = reinterpret_tensor(buf167, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf167  # reuse
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf182, buf184, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf185 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf183, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf184, (48, 64, 17), (1088, 17, 1), 0), out=buf185)
        buf188 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_19], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf185, buf188, 816, 17, grid=grid(816), stream=stream0)
        buf189 = buf185; del buf185  # reuse
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf188, buf189, 13872, grid=grid(13872), stream=stream0)
        buf190 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf182, buf190, 52224, grid=grid(52224), stream=stream0)
        buf191 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf189, reinterpret_tensor(buf190, (48, 17, 64), (1088, 64, 1), 0), out=buf191)
        buf192 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf191, buf192, 52224, grid=grid(52224), stream=stream0)
        buf193 = reinterpret_tensor(buf191, (68, 768), (768, 1), 0); del buf191  # reuse
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf192, (68, 768), (768, 1), 0), reinterpret_tensor(primals_75, (768, 768), (1, 768), 0), out=buf193)
        buf197 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf198 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf349 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_67, layer_norm_13], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf176, buf193, primals_76, primals_77, primals_78, buf197, buf198, buf349, 68, 768, grid=grid(68), stream=stream0)
        del primals_78
        buf199 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_80, reinterpret_tensor(buf198, (68, 768), (768, 1), 0), reinterpret_tensor(primals_79, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf199)
        del primals_80
        buf200 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf199, buf200, 208896, grid=grid(208896), stream=stream0)
        buf201 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf200, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_81, (3072, 768), (1, 3072), 0), out=buf201)
        buf202 = buf176; del buf176  # reuse
        buf206 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf207 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf348 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_67, x_73, layer_norm_14], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf202, buf193, primals_76, buf201, primals_82, primals_83, primals_84, buf206, buf207, buf348, 68, 768, grid=grid(68), stream=stream0)
        del primals_76
        del primals_82
        del primals_84
        buf208 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (68, 768), (768, 1), 0), reinterpret_tensor(primals_85, (768, 2304), (1, 768), 0), out=buf208)
        buf209 = reinterpret_tensor(buf201, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf201  # reuse
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf208, buf209, 52224, grid=grid(52224), stream=stream0)
        buf210 = reinterpret_tensor(buf193, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf193  # reuse
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf208, buf210, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf211 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf210, (48, 64, 17), (1088, 17, 1), 0), out=buf211)
        buf214 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_22], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf211, buf214, 816, 17, grid=grid(816), stream=stream0)
        buf215 = buf211; del buf211  # reuse
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf214, buf215, 13872, grid=grid(13872), stream=stream0)
        buf216 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf208, buf216, 52224, grid=grid(52224), stream=stream0)
        buf217 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf215, reinterpret_tensor(buf216, (48, 17, 64), (1088, 64, 1), 0), out=buf217)
        buf218 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf217, buf218, 52224, grid=grid(52224), stream=stream0)
        buf219 = reinterpret_tensor(buf217, (68, 768), (768, 1), 0); del buf217  # reuse
        # Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf218, (68, 768), (768, 1), 0), reinterpret_tensor(primals_86, (768, 768), (1, 768), 0), out=buf219)
        buf223 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf224 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf347 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_77, layer_norm_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf202, buf219, primals_87, primals_88, primals_89, buf223, buf224, buf347, 68, 768, grid=grid(68), stream=stream0)
        del primals_89
        buf225 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_91, reinterpret_tensor(buf224, (68, 768), (768, 1), 0), reinterpret_tensor(primals_90, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf225)
        del primals_91
        buf226 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_79], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf225, buf226, 208896, grid=grid(208896), stream=stream0)
        buf227 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_81], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf226, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_92, (3072, 768), (1, 3072), 0), out=buf227)
        buf228 = buf202; del buf202  # reuse
        buf232 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf233 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf346 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_77, x_83, layer_norm_16], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf228, buf219, primals_87, buf227, primals_93, primals_94, primals_95, buf232, buf233, buf346, 68, 768, grid=grid(68), stream=stream0)
        del primals_87
        del primals_93
        del primals_95
        buf234 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (68, 768), (768, 1), 0), reinterpret_tensor(primals_96, (768, 2304), (1, 768), 0), out=buf234)
        buf235 = reinterpret_tensor(buf227, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf227  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf234, buf235, 52224, grid=grid(52224), stream=stream0)
        buf236 = reinterpret_tensor(buf219, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf219  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf234, buf236, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf237 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf235, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf236, (48, 64, 17), (1088, 17, 1), 0), out=buf237)
        buf240 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_25], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf237, buf240, 816, 17, grid=grid(816), stream=stream0)
        buf241 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf240, buf241, 13872, grid=grid(13872), stream=stream0)
        buf242 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf234, buf242, 52224, grid=grid(52224), stream=stream0)
        buf243 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf241, reinterpret_tensor(buf242, (48, 17, 64), (1088, 64, 1), 0), out=buf243)
        buf244 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf243, buf244, 52224, grid=grid(52224), stream=stream0)
        buf245 = reinterpret_tensor(buf243, (68, 768), (768, 1), 0); del buf243  # reuse
        # Topologically Sorted Source Nodes: [x_85], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf244, (68, 768), (768, 1), 0), reinterpret_tensor(primals_97, (768, 768), (1, 768), 0), out=buf245)
        buf249 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf250 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf345 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_87, layer_norm_17], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf228, buf245, primals_98, primals_99, primals_100, buf249, buf250, buf345, 68, 768, grid=grid(68), stream=stream0)
        del primals_100
        buf251 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_102, reinterpret_tensor(buf250, (68, 768), (768, 1), 0), reinterpret_tensor(primals_101, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf251)
        del primals_102
        buf252 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_89], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf251, buf252, 208896, grid=grid(208896), stream=stream0)
        buf253 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_91], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf252, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_103, (3072, 768), (1, 3072), 0), out=buf253)
        buf254 = buf228; del buf228  # reuse
        buf258 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf259 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf344 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_87, x_93, x_124, layer_norm_19], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf254, buf245, primals_98, buf253, primals_104, primals_107, primals_108, buf258, buf259, buf344, 68, 768, grid=grid(68), stream=stream0)
        del primals_104
        del primals_108
        del primals_98
        buf260 = buf234; del buf234  # reuse
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf259, (68, 768), (768, 1), 0), reinterpret_tensor(primals_109, (768, 2304), (1, 768), 0), out=buf260)
        buf261 = reinterpret_tensor(buf253, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf260, buf261, 52224, grid=grid(52224), stream=stream0)
        buf262 = reinterpret_tensor(buf245, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf245  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf260, buf262, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf263 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf261, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf262, (48, 64, 17), (1088, 17, 1), 0), out=buf263)
        buf266 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_28], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf263, buf266, 816, 17, grid=grid(816), stream=stream0)
        buf267 = buf263; del buf263  # reuse
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf266, buf267, 13872, grid=grid(13872), stream=stream0)
        buf268 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf260, buf268, 52224, grid=grid(52224), stream=stream0)
        buf269 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf267, reinterpret_tensor(buf268, (48, 17, 64), (1088, 64, 1), 0), out=buf269)
        buf270 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf269, buf270, 52224, grid=grid(52224), stream=stream0)
        buf271 = reinterpret_tensor(buf269, (68, 768), (768, 1), 0); del buf269  # reuse
        # Topologically Sorted Source Nodes: [x_95], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf270, (68, 768), (768, 1), 0), reinterpret_tensor(primals_110, (768, 768), (1, 768), 0), out=buf271)
        buf275 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf276 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf343 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_97, layer_norm_20], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf254, buf271, primals_111, primals_112, primals_113, buf275, buf276, buf343, 68, 768, grid=grid(68), stream=stream0)
        del primals_113
        buf277 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_115, reinterpret_tensor(buf276, (68, 768), (768, 1), 0), reinterpret_tensor(primals_114, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf277)
        del primals_115
        buf278 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf277, buf278, 208896, grid=grid(208896), stream=stream0)
        buf279 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf278, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_116, (3072, 768), (1, 3072), 0), out=buf279)
        buf280 = buf254; del buf254  # reuse
        buf284 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf285 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf342 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_97, x_103, x_125, layer_norm_22], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf280, buf271, primals_111, buf279, primals_117, primals_118, primals_119, buf284, buf285, buf342, 68, 768, grid=grid(68), stream=stream0)
        del primals_111
        del primals_117
        del primals_119
        buf286 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf285, (68, 768), (768, 1), 0), reinterpret_tensor(primals_120, (768, 2304), (1, 768), 0), out=buf286)
        buf287 = reinterpret_tensor(buf279, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf279  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf286, buf287, 52224, grid=grid(52224), stream=stream0)
        buf288 = reinterpret_tensor(buf271, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf271  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf286, buf288, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf289 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf287, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf288, (48, 64, 17), (1088, 17, 1), 0), out=buf289)
        buf292 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_31], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf289, buf292, 816, 17, grid=grid(816), stream=stream0)
        buf293 = buf289; del buf289  # reuse
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf292, buf293, 13872, grid=grid(13872), stream=stream0)
        buf294 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf286, buf294, 52224, grid=grid(52224), stream=stream0)
        buf295 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf293, reinterpret_tensor(buf294, (48, 17, 64), (1088, 64, 1), 0), out=buf295)
        buf296 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf295, buf296, 52224, grid=grid(52224), stream=stream0)
        buf297 = reinterpret_tensor(buf295, (68, 768), (768, 1), 0); del buf295  # reuse
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf296, (68, 768), (768, 1), 0), reinterpret_tensor(primals_121, (768, 768), (1, 768), 0), out=buf297)
        buf301 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf302 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf341 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_107, layer_norm_23], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf280, buf297, primals_122, primals_123, primals_124, buf301, buf302, buf341, 68, 768, grid=grid(68), stream=stream0)
        del primals_124
        buf303 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_126, reinterpret_tensor(buf302, (68, 768), (768, 1), 0), reinterpret_tensor(primals_125, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf303)
        del primals_126
        buf304 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_109], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf303, buf304, 208896, grid=grid(208896), stream=stream0)
        buf305 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf304, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_127, (3072, 768), (1, 3072), 0), out=buf305)
        buf306 = buf280; del buf280  # reuse
        buf310 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf311 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf340 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_107, x_113, x_126, layer_norm_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_17.run(buf306, buf297, primals_122, buf305, primals_128, primals_129, primals_130, buf310, buf311, buf340, 68, 768, grid=grid(68), stream=stream0)
        del primals_122
        del primals_128
        del primals_130
        buf312 = buf286; del buf286  # reuse
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (68, 768), (768, 1), 0), reinterpret_tensor(primals_131, (768, 2304), (1, 768), 0), out=buf312)
        buf313 = reinterpret_tensor(buf305, (4, 12, 17, 64), (13056, 1088, 64, 1), 0); del buf305  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf312, buf313, 52224, grid=grid(52224), stream=stream0)
        buf314 = reinterpret_tensor(buf297, (4, 12, 64, 17), (13056, 1088, 17, 1), 0); del buf297  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_8.run(buf312, buf314, 3072, 17, grid=grid(3072, 17), stream=stream0)
        buf315 = buf293; del buf293  # reuse
        # Topologically Sorted Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf313, (48, 17, 64), (1088, 64, 1), 0), reinterpret_tensor(buf314, (48, 64, 17), (1088, 17, 1), 0), out=buf315)
        buf318 = empty_strided_cuda((4, 12, 17, 17), (3488, 289, 17, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_34], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__softmax_9.run(buf315, buf318, 816, 17, grid=grid(816), stream=stream0)
        buf319 = buf315; del buf315  # reuse
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_10.run(buf318, buf319, 13872, grid=grid(13872), stream=stream0)
        buf320 = empty_strided_cuda((4, 12, 17, 64), (13056, 1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_11.run(buf312, buf320, 52224, grid=grid(52224), stream=stream0)
        del buf312
        buf321 = empty_strided_cuda((48, 17, 64), (1088, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf319, reinterpret_tensor(buf320, (48, 17, 64), (1088, 64, 1), 0), out=buf321)
        del buf319
        buf322 = empty_strided_cuda((4, 17, 12, 64), (13056, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_12.run(buf321, buf322, 52224, grid=grid(52224), stream=stream0)
        buf323 = reinterpret_tensor(buf321, (68, 768), (768, 1), 0); del buf321  # reuse
        # Topologically Sorted Source Nodes: [x_115], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf322, (68, 768), (768, 1), 0), reinterpret_tensor(primals_132, (768, 768), (1, 768), 0), out=buf323)
        buf327 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf328 = empty_strided_cuda((4, 17, 768), (13056, 768, 1), torch.float32)
        buf339 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_117, layer_norm_26], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_16.run(buf306, buf323, primals_133, primals_134, primals_135, buf327, buf328, buf339, 68, 768, grid=grid(68), stream=stream0)
        del primals_135
        buf329 = empty_strided_cuda((68, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_137, reinterpret_tensor(buf328, (68, 768), (768, 1), 0), reinterpret_tensor(primals_136, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf329)
        del primals_137
        buf330 = empty_strided_cuda((4, 17, 3072), (52224, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_14.run(buf329, buf330, 208896, grid=grid(208896), stream=stream0)
        buf331 = empty_strided_cuda((68, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf330, (68, 3072), (3072, 1), 0), reinterpret_tensor(primals_138, (3072, 768), (1, 3072), 0), out=buf331)
        buf332 = buf306; del buf306  # reuse
        buf336 = buf332; del buf332  # reuse
        buf338 = empty_strided_cuda((4, 17, 1), (17, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_117, x_123, x_127], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_18.run(buf336, buf323, primals_133, buf331, primals_139, buf338, 68, 768, grid=grid(68), stream=stream0)
        del buf323
        del buf331
        del primals_133
        del primals_139
        buf337 = reinterpret_tensor(buf19, (4, 3072), (3072, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [embed], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_19.run(buf258, primals_105, primals_106, buf284, buf310, buf336, buf337, 12288, grid=grid(12288), stream=stream0)
        del primals_106
    return (buf337, buf337, primals_1, primals_2, primals_6, primals_11, primals_17, primals_22, primals_28, primals_33, primals_39, primals_44, primals_50, primals_55, primals_61, primals_66, primals_72, primals_77, primals_83, primals_88, primals_94, primals_99, primals_105, primals_107, primals_112, primals_118, primals_123, primals_129, primals_134, buf1, buf2, buf3, buf4, buf5, buf6, buf9, buf12, buf15, buf20, buf21, buf24, reinterpret_tensor(buf25, (68, 768), (768, 1), 0), buf32, reinterpret_tensor(buf36, (68, 768), (768, 1), 0), buf41, reinterpret_tensor(buf42, (68, 768), (768, 1), 0), buf43, reinterpret_tensor(buf44, (68, 3072), (3072, 1), 0), buf50, reinterpret_tensor(buf51, (68, 768), (768, 1), 0), buf58, reinterpret_tensor(buf62, (68, 768), (768, 1), 0), buf67, reinterpret_tensor(buf68, (68, 768), (768, 1), 0), buf69, reinterpret_tensor(buf70, (68, 3072), (3072, 1), 0), buf76, reinterpret_tensor(buf77, (68, 768), (768, 1), 0), buf84, reinterpret_tensor(buf88, (68, 768), (768, 1), 0), buf93, reinterpret_tensor(buf94, (68, 768), (768, 1), 0), buf95, reinterpret_tensor(buf96, (68, 3072), (3072, 1), 0), buf102, reinterpret_tensor(buf103, (68, 768), (768, 1), 0), buf110, reinterpret_tensor(buf114, (68, 768), (768, 1), 0), buf119, reinterpret_tensor(buf120, (68, 768), (768, 1), 0), buf121, reinterpret_tensor(buf122, (68, 3072), (3072, 1), 0), buf128, reinterpret_tensor(buf129, (68, 768), (768, 1), 0), buf136, reinterpret_tensor(buf140, (68, 768), (768, 1), 0), buf145, reinterpret_tensor(buf146, (68, 768), (768, 1), 0), buf147, reinterpret_tensor(buf148, (68, 3072), (3072, 1), 0), buf154, reinterpret_tensor(buf155, (68, 768), (768, 1), 0), buf162, reinterpret_tensor(buf166, (68, 768), (768, 1), 0), buf171, reinterpret_tensor(buf172, (68, 768), (768, 1), 0), buf173, reinterpret_tensor(buf174, (68, 3072), (3072, 1), 0), buf180, reinterpret_tensor(buf181, (68, 768), (768, 1), 0), buf188, reinterpret_tensor(buf192, (68, 768), (768, 1), 0), buf197, reinterpret_tensor(buf198, (68, 768), (768, 1), 0), buf199, reinterpret_tensor(buf200, (68, 3072), (3072, 1), 0), buf206, reinterpret_tensor(buf207, (68, 768), (768, 1), 0), buf214, reinterpret_tensor(buf218, (68, 768), (768, 1), 0), buf223, reinterpret_tensor(buf224, (68, 768), (768, 1), 0), buf225, reinterpret_tensor(buf226, (68, 3072), (3072, 1), 0), buf232, reinterpret_tensor(buf233, (68, 768), (768, 1), 0), buf240, reinterpret_tensor(buf244, (68, 768), (768, 1), 0), buf249, reinterpret_tensor(buf250, (68, 768), (768, 1), 0), buf251, reinterpret_tensor(buf252, (68, 3072), (3072, 1), 0), buf258, reinterpret_tensor(buf259, (68, 768), (768, 1), 0), buf266, reinterpret_tensor(buf270, (68, 768), (768, 1), 0), buf275, reinterpret_tensor(buf276, (68, 768), (768, 1), 0), buf277, reinterpret_tensor(buf278, (68, 3072), (3072, 1), 0), buf284, reinterpret_tensor(buf285, (68, 768), (768, 1), 0), buf292, reinterpret_tensor(buf296, (68, 768), (768, 1), 0), buf301, reinterpret_tensor(buf302, (68, 768), (768, 1), 0), buf303, reinterpret_tensor(buf304, (68, 3072), (3072, 1), 0), buf310, reinterpret_tensor(buf311, (68, 768), (768, 1), 0), buf318, reinterpret_tensor(buf322, (68, 768), (768, 1), 0), buf327, reinterpret_tensor(buf328, (68, 768), (768, 1), 0), buf329, reinterpret_tensor(buf330, (68, 3072), (3072, 1), 0), buf336, buf338, primals_138, primals_136, buf339, primals_132, reinterpret_tensor(buf320, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf313, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf314, (48, 17, 64), (1088, 1, 17), 0), primals_131, buf340, primals_127, primals_125, buf341, primals_121, reinterpret_tensor(buf294, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf287, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf288, (48, 17, 64), (1088, 1, 17), 0), primals_120, buf342, primals_116, primals_114, buf343, primals_110, reinterpret_tensor(buf268, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf261, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf262, (48, 17, 64), (1088, 1, 17), 0), primals_109, buf344, primals_103, primals_101, buf345, primals_97, reinterpret_tensor(buf242, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf235, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf236, (48, 17, 64), (1088, 1, 17), 0), primals_96, buf346, primals_92, primals_90, buf347, primals_86, reinterpret_tensor(buf216, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf209, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf210, (48, 17, 64), (1088, 1, 17), 0), primals_85, buf348, primals_81, primals_79, buf349, primals_75, reinterpret_tensor(buf190, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf183, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf184, (48, 17, 64), (1088, 1, 17), 0), primals_74, buf350, primals_70, primals_68, buf351, primals_64, reinterpret_tensor(buf164, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf157, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf158, (48, 17, 64), (1088, 1, 17), 0), primals_63, buf352, primals_59, primals_57, buf353, primals_53, reinterpret_tensor(buf138, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf131, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf132, (48, 17, 64), (1088, 1, 17), 0), primals_52, buf354, primals_48, primals_46, buf355, primals_42, reinterpret_tensor(buf112, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf105, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf106, (48, 17, 64), (1088, 1, 17), 0), primals_41, buf356, primals_37, primals_35, buf357, primals_31, reinterpret_tensor(buf86, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf79, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf80, (48, 17, 64), (1088, 1, 17), 0), primals_30, buf358, primals_26, primals_24, buf359, primals_20, reinterpret_tensor(buf60, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf53, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf54, (48, 17, 64), (1088, 1, 17), 0), primals_19, buf360, primals_15, primals_13, buf361, primals_9, reinterpret_tensor(buf34, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf27, (48, 64, 17), (1088, 1, 64), 0), reinterpret_tensor(buf28, (48, 17, 64), (1088, 1, 17), 0), primals_8, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((768, 3, 16, 16), (768, 256, 16, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 1, 768), (768, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 197, 768), (151296, 768, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
