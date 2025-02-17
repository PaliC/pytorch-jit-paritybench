# AOT ID: ['34_forward']
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


# kernel path: inductor_cache/3g/c3gfvunhlvkrtq44v5rx5i3fwqajt5lts55ez5jz4v3xnlhskgc7.py
# Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
# Source node to ATen node mapping:
#   pos_embed => clamp_max_2, clamp_min_2, convert_element_type_3, floor_1, sub_4
# Graph fragment:
#   %floor_1 : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%unsqueeze,), kwargs = {})
#   %convert_element_type_3 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor_1, torch.int64), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_3, 1), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_4, 0), kwargs = {})
#   %clamp_max_2 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 13), kwargs = {})
triton_poi_fused__to_copy_clamp_floor_sub_0 = async_compile.triton('triton_poi_fused__to_copy_clamp_floor_sub_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clamp_floor_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clamp_floor_sub_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.875
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


# kernel path: inductor_cache/c4/cc45hvse5skmulw4p5fqvnuj5slnroegqbcrfwsmjo3up3xfabjq.py
# Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   pos_embed => add, clamp_max_5, clamp_min_5, convert_element_type, convert_element_type_2, floor, iota, mul, sub
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.875), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%sub,), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_2, 0), kwargs = {})
#   %clamp_max_5 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 13), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_1 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.875
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


# kernel path: inductor_cache/m4/cm4m2o7yjiyzgjefavlhm6eywmqzglewjrg5kulmvypxvydrqj4h.py
# Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   pos_embed => add, add_4, clamp_max_7, clamp_min_7, convert_element_type, convert_element_type_2, floor, iota, mul, sub
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.875), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%sub,), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 1), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_4, 0), kwargs = {})
#   %clamp_max_7 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 13), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_2 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_2', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.875
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


# kernel path: inductor_cache/kw/ckw3nrnbh653jtibfbyaep67cldtvulxu5df5wq5d4lvzzaiwjlw.py
# Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
# Source node to ATen node mapping:
#   pos_embed => add, add_5, clamp_max_9, clamp_min_9, convert_element_type, convert_element_type_2, floor, iota, mul, sub
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.875), kwargs = {})
#   %sub : [num_users=3] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, 0.5), kwargs = {})
#   %floor : [num_users=2] = call_function[target=torch.ops.aten.floor.default](args = (%sub,), kwargs = {})
#   %convert_element_type_2 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%floor, torch.int64), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_2, 2), kwargs = {})
#   %clamp_min_9 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_5, 0), kwargs = {})
#   %clamp_max_9 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_9, 13), kwargs = {})
triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3 = async_compile.triton('triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 0.875
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


# kernel path: inductor_cache/vx/cvx2hllbyos6rsz7ep5hl3zxckqsmferm236bo6ovdkxwr7zyplu.py
# Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp, aten.rsub, aten._unsafe_index]
# Source node to ATen node mapping:
#   pos_embed => _unsafe_index, _unsafe_index_1, _unsafe_index_10, _unsafe_index_11, _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, _unsafe_index_2, _unsafe_index_3, _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, _unsafe_index_8, _unsafe_index_9, add, add_10, add_11, add_12, add_13, add_14, add_15, add_16, add_17, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_30, add_6, add_7, add_8, add_9, clamp_max, clamp_max_1, clamp_min, clamp_min_1, convert_element_type, floor, floor_1, iota, mul, mul_10, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_3, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_4, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_5, mul_6, mul_7, mul_8, mul_9, sub, sub_10, sub_11, sub_12, sub_13, sub_14, sub_15, sub_16, sub_17, sub_18, sub_19, sub_2, sub_20, sub_21, sub_3, sub_6, sub_7, sub_8, sub_9
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.5), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, 0.875), kwargs = {})
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
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_2, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_2, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_2, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_3 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_2, %clamp_max_9]), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index, %sub_7), kwargs = {})
#   %mul_27 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_1, %add_8), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %mul_27), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_2, %add_9), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %mul_28), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_3, %sub_13), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %mul_29), kwargs = {})
#   %_unsafe_index_4 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_10, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_10, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_6 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_10, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_7 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_10, %clamp_max_9]), kwargs = {})
#   %mul_30 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_4, %sub_7), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_5, %add_8), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_30, %mul_31), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_6, %add_9), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %mul_32), kwargs = {})
#   %mul_33 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_7, %sub_13), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_20, %mul_33), kwargs = {})
#   %_unsafe_index_8 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_18, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_9 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_18, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_10 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_18, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_11 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_18, %clamp_max_9]), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_8, %sub_7), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_9, %add_8), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_34, %mul_35), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_10, %add_9), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %mul_36), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%_unsafe_index_11, %sub_13), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %mul_37), kwargs = {})
#   %_unsafe_index_12 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_26, %clamp_max_3]), kwargs = {})
#   %_unsafe_index_13 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_26, %clamp_max_5]), kwargs = {})
#   %_unsafe_index_14 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_26, %clamp_max_7]), kwargs = {})
#   %_unsafe_index_15 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%primals_5, [None, None, %clamp_max_26, %clamp_max_9]), kwargs = {})
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
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*i64', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*i64', 'in_ptr8': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 16) % 16)
    x0 = (xindex % 16)
    x2 = xindex // 256
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
    tmp9 = tl.load(in_ptr2 + (tmp8 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
    tmp10 = x0
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.5
    tmp13 = tmp11 + tmp12
    tmp14 = 0.875
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
    tmp39 = tl.load(in_ptr2 + (tmp38 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
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
    tmp53 = tl.load(in_ptr2 + (tmp52 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
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
    tmp66 = tl.load(in_ptr2 + (tmp65 + 14*tmp4 + 196*x2), None, eviction_policy='evict_last')
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
    tmp81 = tl.load(in_ptr2 + (tmp8 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp82 = tmp81 * tmp33
    tmp83 = tl.load(in_ptr2 + (tmp38 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp84 = tmp83 * tmp46
    tmp85 = tmp82 + tmp84
    tmp86 = tl.load(in_ptr2 + (tmp52 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp87 = tmp86 * tmp59
    tmp88 = tmp85 + tmp87
    tmp89 = tl.load(in_ptr2 + (tmp65 + 14*tmp80 + 196*x2), None, eviction_policy='evict_last')
    tmp90 = tmp89 * tmp74
    tmp91 = tmp88 + tmp90
    tmp93 = tmp92 + tmp1
    tmp94 = tmp92 < 0
    tmp95 = tl.where(tmp94, tmp93, tmp92)
    tmp96 = tl.load(in_ptr2 + (tmp8 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp97 = tmp96 * tmp33
    tmp98 = tl.load(in_ptr2 + (tmp38 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp99 = tmp98 * tmp46
    tmp100 = tmp97 + tmp99
    tmp101 = tl.load(in_ptr2 + (tmp52 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp102 = tmp101 * tmp59
    tmp103 = tmp100 + tmp102
    tmp104 = tl.load(in_ptr2 + (tmp65 + 14*tmp95 + 196*x2), None, eviction_policy='evict_last')
    tmp105 = tmp104 * tmp74
    tmp106 = tmp103 + tmp105
    tmp108 = tmp107 + tmp1
    tmp109 = tmp107 < 0
    tmp110 = tl.where(tmp109, tmp108, tmp107)
    tmp111 = tl.load(in_ptr2 + (tmp8 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
    tmp112 = tmp111 * tmp33
    tmp113 = tl.load(in_ptr2 + (tmp38 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
    tmp114 = tmp113 * tmp46
    tmp115 = tmp112 + tmp114
    tmp116 = tl.load(in_ptr2 + (tmp52 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
    tmp117 = tmp116 * tmp59
    tmp118 = tmp115 + tmp117
    tmp119 = tl.load(in_ptr2 + (tmp65 + 14*tmp110 + 196*x2), None, eviction_policy='evict_last')
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


# kernel path: inductor_cache/ca/ccauazzyyzi2wjjvhhm3msbrwnbtdkvbx22n4hkcws4leyd6krca.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_2 => add_32
#   x_3 => add_33, clone, rsqrt, var_mean
# Graph fragment:
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %permute_1), kwargs = {})
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_32,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_33,), kwargs = {})
#   %div_47 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 96), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_5 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x2 = xindex // 256
    x5 = (xindex % 256)
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    tmp8_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x5 + 256*r3 + 24576*x2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x5 + 256*r3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr3 + (8*((x1 % 8)) + 64*r3 + ((x0 % 8))), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(rmask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(rmask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(rmask & xmask, tmp8_weight_next, tmp8_weight)
    tmp8_tmp, tmp9_tmp, tmp10_tmp = triton_helpers.welford(
        tmp8_mean, tmp8_m2, tmp8_weight, 1
    )
    tmp8 = tmp8_tmp[:, None]
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp8, xmask)
    tl.store(out_ptr1 + (x4), tmp9, xmask)
    tmp11 = 96.0
    tmp12 = tmp9 / tmp11
    tmp13 = 1e-06
    tmp14 = tmp12 + tmp13
    tmp15 = libdevice.rsqrt(tmp14)
    tmp16 = 0.010416666666666666
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr2 + (x4), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qa/cqagfqrxmk4oz574kcj6lbqhuvtghy34uofns6i45r76zc37svnb.py
# Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_2 => add_32
#   x_3 => add_33, clone, mul_46, rsqrt, sub_22, var_mean
# Graph fragment:
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %permute_1), kwargs = {})
#   %clone : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_32,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_33,), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone, %getitem_1), kwargs = {})
#   %mul_46 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %rsqrt), kwargs = {})
triton_poi_fused_add_native_layer_norm_6 = async_compile.triton('triton_poi_fused_add_native_layer_norm_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    y0 = (yindex % 96)
    x2 = (xindex % 16)
    x3 = xindex // 16
    y1 = yindex // 96
    tmp0 = tl.load(in_ptr0 + (x5 + 256*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x5 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (8*((x3 % 8)) + 64*y0 + ((x2 % 8))), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x5 + 256*y1), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x5 + 256*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 96.0
    tmp11 = tmp9 / tmp10
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (y0 + 96*x5 + 24576*y1), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/nx/cnxdj7gkuyqggjcaqnews3jmdumeu5rfperjscek5mxbtloqxgos.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 768) % 8)
    x3 = ((xindex // 6144) % 2)
    x4 = xindex // 12288
    x5 = (xindex % 768)
    x0 = (xindex % 96)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5 + 768*x3 + 1536*x2 + 12288*x4), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/2s/c2saqzgkixge2ceyzt36gpl54middaeipgy6tvhhngrgdks4pnis.py
# Topologically Sorted Source Nodes: [x_2, x_11], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_11 => add_35
#   x_2 => add_32
# Graph fragment:
#   %add_32 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %permute_1), kwargs = {})
#   %add_35 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_32, %view_9), kwargs = {})
triton_poi_fused_add_8 = async_compile.triton('triton_poi_fused_add_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x5 = xindex
    y4 = yindex
    y0 = (yindex % 96)
    x2 = (xindex % 16)
    x3 = xindex // 16
    y1 = yindex // 96
    tmp0 = tl.load(in_out_ptr0 + (x5 + 256*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x5 + 256*y0), xmask & ymask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (8*((x3 % 8)) + 64*y0 + ((x2 % 8))), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (y0 + 96*((x2 % 8)) + 768*((x3 % 8)) + 6144*(x2 // 8) + 12288*(x3 // 8) + 24576*y1), xmask & ymask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x5 + 256*y4), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kd/ckdvdc3fspc5cmcwg2oghcck65neoxyvgeiwtbnt5l7jigpibrri.py
# Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_1 => add_36, add_37, clone_3, mul_48, mul_49, rsqrt_1, sub_23, var_mean_1
# Graph fragment:
#   %clone_3 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_35,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_3, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_9, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_36,), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_3, %getitem_10), kwargs = {})
#   %mul_48 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %rsqrt_1), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %primals_12), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %primals_13), kwargs = {})
#   %div_46 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 96), kwargs = {})
triton_red_fused_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_red_fused_native_layer_norm_native_layer_norm_backward_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_native_layer_norm_native_layer_norm_backward_9(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r2 + 24576*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight, roffset == 0
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + (x0 + 256*r2 + 24576*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp13 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 96.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-06
        tmp10 = tmp8 + tmp9
        tmp11 = libdevice.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = tmp12 * tmp13
        tmp16 = tmp14 + tmp15
        tl.store(out_ptr2 + (r2 + 96*x3), tmp12, rmask & xmask)
        tl.store(out_ptr3 + (r2 + 96*x3), tmp16, rmask & xmask)
    tmp17 = 96.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-06
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = 0.010416666666666666
    tmp23 = tmp21 * tmp22
    tl.store(out_ptr4 + (x3), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2j/c2jxqogfpynmzlddb2tmbyarspd5j72m5z3lupk6dyhhecvhx2hy.py
# Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_12 => add_38, erf, mul_50, mul_51, mul_52
# Graph fragment:
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, 0.5), kwargs = {})
#   %mul_51 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_11, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_51,), kwargs = {})
#   %add_38 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_50, %add_38), kwargs = {})
triton_poi_fused_gelu_10 = async_compile.triton('triton_poi_fused_gelu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
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


# kernel path: inductor_cache/zq/czq4yd62bffstvtuxtglrlqeukv3ra2an27t7rdrzi2xxctvfer3.py
# Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_14 => add_39
#   x_15 => add_40, clone_4, mul_53, rsqrt_2, sub_24, var_mean_2
# Graph fragment:
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %view_13), kwargs = {})
#   %clone_4 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_39,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_11, 1e-06), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_40,), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_4, %getitem_12), kwargs = {})
#   %mul_53 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %rsqrt_2), kwargs = {})
#   %div_45 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 96), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r2 + 24576*x1), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + 96*x3), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp9 = tl.load(in_ptr0 + (x0 + 256*r2 + 24576*x1), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp10 = tl.load(in_ptr1 + (r2 + 96*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp9 + tmp12
        tmp14 = tmp13 - tmp6
        tmp15 = 96.0
        tmp16 = tmp7 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tl.store(out_ptr2 + (r2 + 96*x3), tmp20, rmask & xmask)
    tmp21 = 96.0
    tmp22 = tmp7 / tmp21
    tmp23 = 1e-06
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.rsqrt(tmp24)
    tmp26 = 0.010416666666666666
    tmp27 = tmp25 * tmp26
    tl.store(out_ptr3 + (x3), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jx/cjxwqvqcclhynjvdhdgoqxxosqxlt6fzk3ya3wfw4czbetpzmvol.py
# Topologically Sorted Source Nodes: [x_14, x_23, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_3 => add_43, add_44, clone_7, mul_55, mul_56, rsqrt_3, sub_25, var_mean_3
#   x_14 => add_39
#   x_23 => add_42
# Graph fragment:
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_35, %view_13), kwargs = {})
#   %add_42 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %view_23), kwargs = {})
#   %clone_7 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_42,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_7, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_43,), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_7, %getitem_21), kwargs = {})
#   %mul_55 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %rsqrt_3), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %primals_24), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %primals_25), kwargs = {})
#   %div_44 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 96), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x2 = xindex // 256
    x5 = (xindex % 256)
    x4 = xindex
    x0 = (xindex % 16)
    x1 = ((xindex // 16) % 16)
    tmp10_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x5 + 256*r3 + 24576*x2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r3 + 96*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r3 + 96*((x0 % 8)) + 768*((x1 % 8)) + 6144*(x0 // 8) + 12288*(x1 // 8) + 24576*x2), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(rmask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(rmask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(rmask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r3 + 96*x4), tmp8, rmask & xmask)
    tmp10_tmp, tmp11_tmp, tmp12_tmp = triton_helpers.welford(
        tmp10_mean, tmp10_m2, tmp10_weight, 1
    )
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp13 = tl.load(in_out_ptr0 + (r3 + 96*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp21 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp23 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tmp13 - tmp10
        tmp15 = 96.0
        tmp16 = tmp11 / tmp15
        tmp17 = 1e-06
        tmp18 = tmp16 + tmp17
        tmp19 = libdevice.rsqrt(tmp18)
        tmp20 = tmp14 * tmp19
        tmp22 = tmp20 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r3 + 96*x4), tmp20, rmask & xmask)
        tl.store(out_ptr3 + (r3 + 96*x4), tmp24, rmask & xmask)
    tmp25 = 96.0
    tmp26 = tmp11 / tmp25
    tmp27 = 1e-06
    tmp28 = tmp26 + tmp27
    tmp29 = libdevice.rsqrt(tmp28)
    tmp30 = 0.010416666666666666
    tmp31 = tmp29 * tmp30
    tl.store(out_ptr4 + (x4), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7d/c7dea26b4zzczuuaqlsgwhj7eloag26ymcuissvjp564xtccycgp.py
# Topologically Sorted Source Nodes: [x_26, x_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_26 => add_46
#   x_27 => add_47, add_48, clone_8, mul_60, mul_61, rsqrt_4, sub_26, var_mean_4
# Graph fragment:
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_42, %view_27), kwargs = {})
#   %clone_8 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_46,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_8, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_22, 1e-06), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_47,), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_8, %getitem_23), kwargs = {})
#   %mul_60 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %rsqrt_4), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_60, %primals_30), kwargs = {})
#   %add_48 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_61, %primals_31), kwargs = {})
#   %div_43 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 96), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 96
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 96*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 96*x0), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 96, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 96.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.010416666666666666
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r1 + 96*x0), tmp4, rmask & xmask)
    tl.store(out_ptr2 + (r1 + 96*x0), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + 96*x0), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2yyz7b2tskydoii5vko5rusgmg23p2k7e6thqd2s54dos5ccfh.py
# Topologically Sorted Source Nodes: [x_26, feats], Original ATen: [aten.add, aten.permute]
# Source node to ATen node mapping:
#   feats => permute_22
#   x_26 => add_46
# Graph fragment:
#   %add_46 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_42, %view_27), kwargs = {})
#   %permute_22 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add_46, [0, 3, 1, 2]), kwargs = {})
triton_poi_fused_add_permute_14 = async_compile.triton('triton_poi_fused_add_permute_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 256}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_permute_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_permute_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 24576*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 256*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/7i/c7izxpbdu4yiffkencky6n7huy45bmkqotycagvxwkelc22solhe.py
# Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_4 => clone_9
# Graph fragment:
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_26,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_15 = async_compile.triton('triton_poi_fused_clone_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 8)
    x2 = ((xindex // 6144) % 2)
    x3 = xindex // 12288
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x2 + 1536*x1 + 12288*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/lw/clwb2ep4muew4ag5r32dxbbpyeybyb5nuoqiqalgsgzzies5lyro.py
# Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_33 => _low_memory_max_pool2d_with_offsets_1, getitem_30
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%permute_28, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_30 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_16 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_16(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 4)
    x2 = xindex // 768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1152*x1 + 9216*x2), None)
    tmp1 = tl.load(in_ptr0 + (576 + x0 + 1152*x1 + 9216*x2), None)
    tmp7 = tl.load(in_ptr0 + (4608 + x0 + 1152*x1 + 9216*x2), None)
    tmp12 = tl.load(in_ptr0 + (5184 + x0 + 1152*x1 + 9216*x2), None)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/nb/cnb6swrl5u6xr6v4yedkgbj2nwr2o7c3wglm37feu2iyijhfizwf.py
# Topologically Sorted Source Nodes: [x_29, x_41, layer_norm_5], Original ATen: [aten.max_pool2d_with_indices, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_5 => add_50, add_51, mul_62, mul_63, rsqrt_5, sub_27, var_mean_5
#   x_29 => getitem_25
#   x_41 => add_49
# Graph fragment:
#   %getitem_25 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %add_49 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_25, %view_41), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_49, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_35, 1e-06), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_50,), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_49, %getitem_36), kwargs = {})
#   %mul_62 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %rsqrt_5), kwargs = {})
#   %mul_63 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_62, %primals_38), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_63, %primals_39), kwargs = {})
#   %div_42 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 192), kwargs = {})
triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_17 = async_compile.triton('triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 8)
    x1 = xindex // 8
    x5 = xindex
    x3 = ((xindex // 8) % 8)
    x4 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (r2 + 384*x0 + 6144*x1), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (192 + r2 + 384*x0 + 6144*x1), rmask & xmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (3072 + r2 + 384*x0 + 6144*x1), rmask & xmask, other=0.0)
    tmp12 = tl.load(in_ptr0 + (3264 + r2 + 384*x0 + 6144*x1), rmask & xmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (r2 + 192*((x0 % 4)) + 768*((x3 % 4)) + 3072*(x0 // 4) + 6144*(x3 // 4) + 12288*x4), rmask & xmask, other=0.0)
    tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1, 1], 1, tl.int8)
    tmp4 = tl.full([1, 1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1, 1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1, 1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    tmp26 = tl.where(rmask & xmask, tmp24, 0)
    tmp27 = tl.sum(tmp26, 1)[:, None]
    tmp28 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK, RBLOCK])
    tmp35 = tl.where(rmask & xmask, tmp33, 0)
    tmp36 = tl.sum(tmp35, 1)[:, None]
    tmp37 = tmp20 - tmp30
    tmp38 = 192.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = 0.005208333333333333
    tmp49 = tmp42 * tmp48
    tl.store(out_ptr0 + (r2 + 192*x5), tmp15, rmask & xmask)
    tl.store(out_ptr1 + (r2 + 192*x5), tmp20, rmask & xmask)
    tl.store(out_ptr4 + (r2 + 192*x5), tmp43, rmask & xmask)
    tl.store(out_ptr5 + (r2 + 192*x5), tmp47, rmask & xmask)
    tl.store(out_ptr6 + (x5), tmp49, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ux/cux4csj4a3moth5qkpiet7lhlv4vrifpasvxrl52n7noyjrvr2ba.py
# Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_42 => add_52, erf_2, mul_64, mul_65, mul_66
# Graph fragment:
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_43, 0.5), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_43, 0.7071067811865476), kwargs = {})
#   %erf_2 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_65,), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_2, 1), kwargs = {})
#   %mul_66 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %add_52), kwargs = {})
triton_poi_fused_gelu_18 = async_compile.triton('triton_poi_fused_gelu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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


# kernel path: inductor_cache/4x/c4xwh7qmk4thugvevh6mey26vsavrskqbbqb7irfvtiicv7vru2t.py
# Topologically Sorted Source Nodes: [x_44, x_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_44 => add_53
#   x_45 => add_54, mul_67, rsqrt_6, sub_28, var_mean_6
# Graph fragment:
#   %add_53 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %view_45), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_53, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_54 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_37, 1e-06), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_54,), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_53, %getitem_38), kwargs = {})
#   %mul_67 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %rsqrt_6), kwargs = {})
#   %div_41 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_6, 192), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 192*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 192*x0), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp28 = 0.005208333333333333
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr2 + (r1 + 192*x0), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp29, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jk/cjkjg7hgwluwvx3olocdvslrvyjwp4e4rm5bi6t5wxyzopp3xwg2.py
# Topologically Sorted Source Nodes: [contiguous_6], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_6 => clone_11
# Graph fragment:
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_38,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 768) % 4)
    x3 = ((xindex // 3072) % 2)
    x4 = xindex // 6144
    x5 = (xindex % 768)
    x0 = (xindex % 192)
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5 + 768*x3 + 1536*x2 + 6144*x4), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x6), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/7i/c7istjaymt7hqrdmimyn3l5m42awxkfo5ze265driy5x7zraxnb3.py
# Topologically Sorted Source Nodes: [x_44, x_53, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_7 => add_57, add_58, mul_69, mul_70, rsqrt_7, sub_29, var_mean_7
#   x_44 => add_53
#   x_53 => add_56
# Graph fragment:
#   %add_53 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_49, %view_45), kwargs = {})
#   %add_56 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_53, %view_55), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_56, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_46, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_57,), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_56, %getitem_47), kwargs = {})
#   %mul_69 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %rsqrt_7), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_69, %primals_50), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, %primals_51), kwargs = {})
#   %div_40 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_7, 192), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = (xindex % 8)
    x1 = ((xindex // 8) % 8)
    x2 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (r3 + 192*x4), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r3 + 192*x4), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r3 + 192*((x0 % 4)) + 768*((x1 % 4)) + 3072*(x0 // 4) + 6144*(x1 // 4) + 12288*x2), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.005208333333333333
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r3 + 192*x4), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r3 + 192*x4), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r3 + 192*x4), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x4), tmp37, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bt/cbtqq4iyxsbecepjp3dpo65bqy22r34co43efipuzfzx6u4ktlug.py
# Topologically Sorted Source Nodes: [x_68, x_69], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_68 => add_67
#   x_69 => add_68, add_69, mul_81, mul_82, rsqrt_10, sub_32, var_mean_10
# Graph fragment:
#   %add_67 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_63, %view_73), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_67, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_68 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_59, 1e-06), kwargs = {})
#   %rsqrt_10 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_68,), kwargs = {})
#   %sub_32 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_67, %getitem_60), kwargs = {})
#   %mul_81 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_32, %rsqrt_10), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_81, %primals_68), kwargs = {})
#   %add_69 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_82, %primals_69), kwargs = {})
#   %div_37 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_10, 192), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 192*x0), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 192*x0), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.005208333333333333
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r1 + 192*x0), tmp4, rmask & xmask)
    tl.store(out_ptr2 + (r1 + 192*x0), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + 192*x0), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nv/cnvo5ywqocj5b6rjqexcdztf4qzk36gnahecs6sdlwh4cleq4u2p.py
# Topologically Sorted Source Nodes: [contiguous_10], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous_10 => clone_15
# Graph fragment:
#   %clone_15 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_62,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_23 = async_compile.triton('triton_poi_fused_clone_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 4)
    x2 = ((xindex // 3072) % 2)
    x3 = xindex // 6144
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768*x2 + 1536*x1 + 6144*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/e3/ce3aaq4gh5cux7tm5cjr6g7fbxyrwzhpwiaeipks4etqg6maixmp.py
# Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_75 => _low_memory_max_pool2d_with_offsets_3, getitem_67
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%permute_64, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_67 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_24 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_24(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 2)
    x2 = xindex // 768
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 2304*x1 + 9216*x2), None)
    tmp1 = tl.load(in_ptr0 + (1152 + x0 + 2304*x1 + 9216*x2), None)
    tmp7 = tl.load(in_ptr0 + (4608 + x0 + 2304*x1 + 9216*x2), None)
    tmp12 = tl.load(in_ptr0 + (5760 + x0 + 2304*x1 + 9216*x2), None)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: inductor_cache/yo/cyot3xuihcbnuf3hoj7rtg47vrhe7ryfrckodvrl43fsgwhxdlfi.py
# Topologically Sorted Source Nodes: [x_71, x_83, layer_norm_11], Original ATen: [aten.max_pool2d_with_indices, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_11 => add_71, add_72, mul_83, mul_84, rsqrt_11, sub_33, var_mean_11
#   x_71 => getitem_62
#   x_83 => add_70
# Graph fragment:
#   %getitem_62 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
#   %add_70 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_61, %view_87), kwargs = {})
#   %var_mean_11 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_70, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_72, 1e-06), kwargs = {})
#   %rsqrt_11 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_71,), kwargs = {})
#   %sub_33 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_70, %getitem_73), kwargs = {})
#   %mul_83 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_33, %rsqrt_11), kwargs = {})
#   %mul_84 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_83, %primals_76), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_84, %primals_77), kwargs = {})
#   %div_36 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_11, 384), kwargs = {})
triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_25 = async_compile.triton('triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r2 = rindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    x5 = xindex
    x3 = ((xindex // 4) % 4)
    x4 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (r2 + 768*x0 + 6144*x1), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (384 + r2 + 768*x0 + 6144*x1), rmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (3072 + r2 + 768*x0 + 6144*x1), rmask, other=0.0)
    tmp12 = tl.load(in_ptr0 + (3456 + r2 + 768*x0 + 6144*x1), rmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (r2 + 384*((x0 % 2)) + 768*((x3 % 2)) + 1536*(x0 // 2) + 3072*(x3 // 2) + 6144*x4), rmask, other=0.0)
    tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 384, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 384.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = 0.0026041666666666665
    tmp49 = tmp42 * tmp48
    tl.store(out_ptr0 + (r2 + 384*x5), tmp15, rmask)
    tl.store(out_ptr1 + (r2 + 384*x5), tmp20, rmask)
    tl.store(out_ptr4 + (r2 + 384*x5), tmp43, rmask)
    tl.store(out_ptr5 + (r2 + 384*x5), tmp47, rmask)
    tl.store(out_ptr6 + (x5), tmp49, None)
''', device_str='cuda')


# kernel path: inductor_cache/wk/cwkhce5tvwjm3tjkqwd6ngx7cf4boch66tlqzizglq5m6l4vydua.py
# Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_84 => add_73, erf_5, mul_85, mul_86, mul_87
# Graph fragment:
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_89, 0.5), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_89, 0.7071067811865476), kwargs = {})
#   %erf_5 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_86,), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_5, 1), kwargs = {})
#   %mul_87 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %add_73), kwargs = {})
triton_poi_fused_gelu_26 = async_compile.triton('triton_poi_fused_gelu_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
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


# kernel path: inductor_cache/sn/csnfotk6xb5ptjvz27hmw22btzpbsvj2p6ihmk534im4xllarzk5.py
# Topologically Sorted Source Nodes: [x_86, x_87], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_86 => add_74
#   x_87 => add_75, mul_88, rsqrt_12, sub_34, var_mean_12
# Graph fragment:
#   %add_74 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %view_91), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_74, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_75 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_74, 1e-06), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_75,), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_74, %getitem_75), kwargs = {})
#   %mul_88 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %rsqrt_12), kwargs = {})
#   %div_35 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_12, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 384*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp28 = 0.0026041666666666665
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr2 + (r1 + 384*x0), tmp27, rmask)
    tl.store(out_ptr3 + (x0), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/3u/c3uzwmctqb3kj25jdm3qeo5dtkjuz3r4n22i4ewx5qj3vbll5zc2.py
# Topologically Sorted Source Nodes: [x_87, x_88], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_87 => add_76, mul_89
#   x_88 => constant_pad_nd
# Graph fragment:
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_88, %primals_82), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_89, %primals_83), kwargs = {})
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_76, [0, 0, 0, 10, 0, 10], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_native_layer_norm_28 = async_compile.triton('triton_poi_fused_constant_pad_nd_native_layer_norm_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_native_layer_norm_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_native_layer_norm_28(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 5376) % 14)
    x1 = ((xindex // 384) % 14)
    x3 = xindex // 75264
    x4 = (xindex % 5376)
    x0 = (xindex % 384)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 1536*x2 + 6144*x3), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr2 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tl.store(out_ptr0 + (x5), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6n/c6nxpgehh4goptqc3x4dnc5yukuc7irhxix37ygynitbkq3ulxzs.py
# Topologically Sorted Source Nodes: [x_86, x_96, x_97, layer_norm_13], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_13 => add_78, add_79, mul_90, mul_91, rsqrt_13, sub_35, var_mean_13
#   x_86 => add_74
#   x_96 => clone_17
#   x_97 => add_77
# Graph fragment:
#   %add_74 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_70, %view_91), kwargs = {})
#   %clone_17 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_3,), kwargs = {memory_format: torch.contiguous_format})
#   %add_77 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_74, %clone_17), kwargs = {})
#   %var_mean_13 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_77, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_78 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_83, 1e-06), kwargs = {})
#   %rsqrt_13 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_78,), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_77, %getitem_84), kwargs = {})
#   %mul_90 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %rsqrt_13), kwargs = {})
#   %mul_91 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_90, %primals_88), kwargs = {})
#   %add_79 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_91, %primals_89), kwargs = {})
#   %div_34 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_13, 384), kwargs = {})
triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_out_ptr0 + (r3 + 384*x4), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r3 + 384*x4), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r3 + 384*x0 + 5376*x1 + 75264*x2), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0026041666666666665
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r3 + 384*x4), tmp8, rmask)
    tl.store(out_ptr2 + (r3 + 384*x4), tmp31, rmask)
    tl.store(out_ptr3 + (r3 + 384*x4), tmp35, rmask)
    tl.store(out_ptr4 + (x4), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/nk/cnkosglsnuiq7hieeyljgza4dc2u6jw3y7rxtfelf4j7pppsj4wd.py
# Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_170 => add_116
#   x_171 => add_117, add_118, mul_130, mul_131, rsqrt_24, sub_46, var_mean_24
# Graph fragment:
#   %add_116 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_112, %view_175), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_116, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_117 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_140, 1e-06), kwargs = {})
#   %rsqrt_24 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_117,), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_116, %getitem_141), kwargs = {})
#   %mul_130 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %rsqrt_24), kwargs = {})
#   %mul_131 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_130, %primals_154), kwargs = {})
#   %add_118 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_131, %primals_155), kwargs = {})
#   %div_23 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_24, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 384*x0), rmask, other=0.0)
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
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0026041666666666665
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r1 + 384*x0), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + 384*x0), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/qn/cqnngqsr7rmd7bsbrdxjxkz4csjoxbdfrsiytspog2cjr2pf67fc.py
# Topologically Sorted Source Nodes: [x_170, x_176, layer_norm_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_25 => add_120, add_121, mul_132, mul_133, rsqrt_25, sub_47, var_mean_25
#   x_170 => add_116
#   x_176 => add_119
# Graph fragment:
#   %add_116 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_112, %view_175), kwargs = {})
#   %add_119 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_116, %view_181), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_119, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_120 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_149, 1e-06), kwargs = {})
#   %rsqrt_25 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_120,), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_119, %getitem_150), kwargs = {})
#   %mul_132 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %rsqrt_25), kwargs = {})
#   %mul_133 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_132, %primals_160), kwargs = {})
#   %add_121 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_133, %primals_161), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_25, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 384*x0), rmask, other=0.0)
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
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0026041666666666665
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r1 + 384*x0), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + 384*x0), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + 384*x0), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/jg/cjgjf6gqckpgq56dabjkxi6ybgltvbqmk5jmw7jnlyxpzx6iamub.py
# Topologically Sorted Source Nodes: [x_281, x_282], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_281 => add_179
#   x_282 => add_180, add_181, mul_193, mul_194, rsqrt_42, sub_64, var_mean_42
# Graph fragment:
#   %add_179 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_175, %view_289), kwargs = {})
#   %var_mean_42 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_179, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_180 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_239, 1e-06), kwargs = {})
#   %rsqrt_42 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_180,), kwargs = {})
#   %sub_64 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_179, %getitem_240), kwargs = {})
#   %mul_193 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_64, %rsqrt_42), kwargs = {})
#   %mul_194 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_193, %primals_262), kwargs = {})
#   %add_181 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_194, %primals_263), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_42, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 64, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0026041666666666665
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r1 + 384*x0), tmp4, rmask)
    tl.store(out_ptr2 + (r1 + 384*x0), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + 384*x0), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/qv/cqvj6qwprfs443ggpnojds5nkbfqdcqjynbsxzh7dsuheytc7mpv.py
# Topologically Sorted Source Nodes: [x_286], Original ATen: [aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_286 => constant_pad_nd_12
# Graph fragment:
#   %constant_pad_nd_12 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_181, [0, 0, 0, 10, 0, 10], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_33 = async_compile.triton('triton_poi_fused_constant_pad_nd_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_33(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 5376) % 14)
    x1 = ((xindex // 384) % 14)
    x3 = xindex // 75264
    x4 = (xindex % 5376)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 4, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 1536*x2 + 6144*x3), tmp5 & xmask, other=0.0)
    tl.store(out_ptr0 + (x5), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wu/cwuxreaviprdwltnswgzd7iqe3kpav4mfxcm6eyv7armzkjfhenl.py
# Topologically Sorted Source Nodes: [x_289], Original ATen: [aten.max_pool2d_with_indices]
# Source node to ATen node mapping:
#   x_289 => _low_memory_max_pool2d_with_offsets_5, getitem_247
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_5 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%permute_224, [2, 2], [2, 2], [0, 0], [1, 1], False), kwargs = {})
#   %getitem_247 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_5, 1), kwargs = {})
triton_poi_fused_max_pool2d_with_indices_34 = async_compile.triton('triton_poi_fused_max_pool2d_with_indices_34', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_34(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 768)
    x1 = ((xindex // 768) % 7)
    x2 = xindex // 5376
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4608*x1 + 64512*x2), xmask)
    tmp1 = tl.load(in_ptr0 + (2304 + x0 + 4608*x1 + 64512*x2), xmask)
    tmp7 = tl.load(in_ptr0 + (32256 + x0 + 4608*x1 + 64512*x2), xmask)
    tmp12 = tl.load(in_ptr0 + (34560 + x0 + 4608*x1 + 64512*x2), xmask)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5x/c5xfxiov4wdkwahtwbci67lyyjkaumfjwl2zmzx27ulu7usqb7f2.py
# Topologically Sorted Source Nodes: [x_284, x_297, x_298, layer_norm_43], Original ATen: [aten.max_pool2d_with_indices, aten.clone, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_43 => add_183, add_184, mul_195, mul_196, rsqrt_43, sub_65, var_mean_43
#   x_284 => getitem_242
#   x_297 => clone_29
#   x_298 => add_182
# Graph fragment:
#   %getitem_242 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_4, 1), kwargs = {})
#   %clone_29 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_51,), kwargs = {memory_format: torch.contiguous_format})
#   %add_182 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_221, %clone_29), kwargs = {})
#   %var_mean_43 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_182, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_183 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_252, 1e-06), kwargs = {})
#   %rsqrt_43 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_183,), kwargs = {})
#   %sub_65 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_182, %getitem_253), kwargs = {})
#   %mul_195 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_65, %rsqrt_43), kwargs = {})
#   %mul_196 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_195, %primals_270), kwargs = {})
#   %add_184 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_196, %primals_271), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_43, 768), kwargs = {})
triton_per_fused_add_clone_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_35 = async_compile.triton('triton_per_fused_add_clone_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr4, out_ptr5, out_ptr6, xnumel, rnumel):
    xnumel = 16
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
    x0 = (xindex % 2)
    x1 = xindex // 2
    x5 = xindex
    x3 = ((xindex // 2) % 2)
    x4 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (r2 + 1536*x0 + 6144*x1), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (768 + r2 + 1536*x0 + 6144*x1), rmask, other=0.0)
    tmp7 = tl.load(in_ptr0 + (3072 + r2 + 1536*x0 + 6144*x1), rmask, other=0.0)
    tmp12 = tl.load(in_ptr0 + (3840 + r2 + 1536*x0 + 6144*x1), rmask, other=0.0)
    tmp17 = tl.load(in_ptr1 + (r2 + 768*x0 + 5376*x3 + 37632*x4), rmask, other=0.0)
    tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1 > tmp0
    tmp3 = tl.full([1], 1, tl.int8)
    tmp4 = tl.full([1], 0, tl.int8)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = triton_helpers.maximum(tmp1, tmp0)
    tmp8 = tmp7 > tmp6
    tmp9 = tl.full([1], 2, tl.int8)
    tmp10 = tl.where(tmp8, tmp9, tmp5)
    tmp11 = triton_helpers.maximum(tmp7, tmp6)
    tmp13 = tmp12 > tmp11
    tmp14 = tl.full([1], 3, tl.int8)
    tmp15 = tl.where(tmp13, tmp14, tmp10)
    tmp16 = triton_helpers.maximum(tmp12, tmp11)
    tmp19 = tmp17 + tmp18
    tmp20 = tmp16 + tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp28 = tl.full([1], 768, tl.int32)
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp27 / tmp29
    tmp31 = tmp21 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = tmp20 - tmp30
    tmp38 = 768.0
    tmp39 = tmp36 / tmp38
    tmp40 = 1e-06
    tmp41 = tmp39 + tmp40
    tmp42 = libdevice.rsqrt(tmp41)
    tmp43 = tmp37 * tmp42
    tmp45 = tmp43 * tmp44
    tmp47 = tmp45 + tmp46
    tmp48 = 0.0013020833333333333
    tmp49 = tmp42 * tmp48
    tl.store(out_ptr0 + (r2 + 768*x5), tmp15, rmask)
    tl.store(out_ptr1 + (r2 + 768*x5), tmp20, rmask)
    tl.store(out_ptr4 + (r2 + 768*x5), tmp43, rmask)
    tl.store(out_ptr5 + (r2 + 768*x5), tmp47, rmask)
    tl.store(out_ptr6 + (x5), tmp49, None)
''', device_str='cuda')


# kernel path: inductor_cache/yg/cygo4jr45xcxdn7gai5cnpbjogkb6rk37gt3pxqqnrbxzlpqsjgf.py
# Topologically Sorted Source Nodes: [x_299], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_299 => add_185, erf_21, mul_197, mul_198, mul_199
# Graph fragment:
#   %mul_197 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_305, 0.5), kwargs = {})
#   %mul_198 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_305, 0.7071067811865476), kwargs = {})
#   %erf_21 : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_198,), kwargs = {})
#   %add_185 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf_21, 1), kwargs = {})
#   %mul_199 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_197, %add_185), kwargs = {})
triton_poi_fused_gelu_36 = async_compile.triton('triton_poi_fused_gelu_36', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_36', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_36(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
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


# kernel path: inductor_cache/nn/cnns5m4ublqtxhokmc3zhdpytsgzzzbs65dwfsbrey3hykgr7wpg.py
# Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   x_301 => add_186
#   x_302 => add_187, mul_200, rsqrt_44, sub_66, var_mean_44
# Graph fragment:
#   %add_186 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_182, %view_307), kwargs = {})
#   %var_mean_44 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_186, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_187 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_254, 1e-06), kwargs = {})
#   %rsqrt_44 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_187,), kwargs = {})
#   %sub_66 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_186, %getitem_255), kwargs = {})
#   %mul_200 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_66, %rsqrt_44), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_44, 768), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_37 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_37', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_37(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 16
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
    tmp24 = 1e-06
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp28 = 0.0013020833333333333
    tmp29 = tmp26 * tmp28
    tl.store(out_ptr2 + (r1 + 768*x0), tmp27, rmask)
    tl.store(out_ptr3 + (x0), tmp29, None)
''', device_str='cuda')


# kernel path: inductor_cache/yj/cyjhsow4uxh7qqbpvnuuab6s3t2cphz2mfisclrjed2hvm3mkhs7.py
# Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
# Source node to ATen node mapping:
#   x_302 => add_188, mul_201
#   x_303 => constant_pad_nd_13
# Graph fragment:
#   %mul_201 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_200, %primals_276), kwargs = {})
#   %add_188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_201, %primals_277), kwargs = {})
#   %constant_pad_nd_13 : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%add_188, [0, 0, 0, 5, 0, 5], 0.0), kwargs = {})
triton_poi_fused_constant_pad_nd_native_layer_norm_38 = async_compile.triton('triton_poi_fused_constant_pad_nd_native_layer_norm_38', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_native_layer_norm_38', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_native_layer_norm_38(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 5376) % 7)
    x1 = ((xindex // 768) % 7)
    x3 = xindex // 37632
    x4 = (xindex % 5376)
    x0 = (xindex % 768)
    x5 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 2, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x1
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x4 + 1536*x2 + 3072*x3), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 * tmp7
    tmp9 = tl.load(in_ptr2 + (x0), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp5, tmp10, tmp11)
    tl.store(out_ptr0 + (x5), tmp12, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ut/cutkm3ukidrmdmsalxm4btnna4oxkbprmlxool32in2lo5jjshgo.py
# Topologically Sorted Source Nodes: [x_301, x_311, x_312, layer_norm_45], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_45 => add_190, add_191, mul_202, mul_203, rsqrt_45, sub_67, var_mean_45
#   x_301 => add_186
#   x_311 => clone_30
#   x_312 => add_189
# Graph fragment:
#   %add_186 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_182, %view_307), kwargs = {})
#   %clone_30 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%slice_55,), kwargs = {memory_format: torch.contiguous_format})
#   %add_189 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_186, %clone_30), kwargs = {})
#   %var_mean_45 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_189, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_263, 1e-06), kwargs = {})
#   %rsqrt_45 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_190,), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_189, %getitem_264), kwargs = {})
#   %mul_202 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %rsqrt_45), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_202, %primals_282), kwargs = {})
#   %add_191 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_203, %primals_283), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_45, 768), kwargs = {})
triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_39 = async_compile.triton('triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_39', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 1024},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = (xindex % 2)
    x1 = ((xindex // 2) % 2)
    x2 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (r3 + 768*x4), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r3 + 768*x4), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r3 + 768*x0 + 5376*x1 + 37632*x2), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r3), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r3), rmask, eviction_policy='evict_last', other=0.0)
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
    tmp28 = 1e-06
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0013020833333333333
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r3 + 768*x4), tmp8, rmask)
    tl.store(out_ptr2 + (r3 + 768*x4), tmp31, rmask)
    tl.store(out_ptr3 + (r3 + 768*x4), tmp35, rmask)
    tl.store(out_ptr4 + (x4), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/lw/clwflil6lrmebkijagqr7tofgprahaz4l3r4ebywr7j7pbyrbmbn.py
# Topologically Sorted Source Nodes: [x_329], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_329 => add_200
# Graph fragment:
#   %add_200 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_196, %view_335), kwargs = {})
triton_poi_fused_add_40 = async_compile.triton('triton_poi_fused_add_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_40(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299 = args
    args.clear()
    assert_size_stride(primals_1, (96, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (96, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (1, 96, 8, 8), (6144, 64, 8, 1))
    assert_size_stride(primals_5, (1, 96, 14, 14), (18816, 196, 14, 1))
    assert_size_stride(primals_6, (96, ), (1, ))
    assert_size_stride(primals_7, (96, ), (1, ))
    assert_size_stride(primals_8, (288, 96), (96, 1))
    assert_size_stride(primals_9, (288, ), (1, ))
    assert_size_stride(primals_10, (96, 96), (96, 1))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_12, (96, ), (1, ))
    assert_size_stride(primals_13, (96, ), (1, ))
    assert_size_stride(primals_14, (384, 96), (96, 1))
    assert_size_stride(primals_15, (384, ), (1, ))
    assert_size_stride(primals_16, (96, 384), (384, 1))
    assert_size_stride(primals_17, (96, ), (1, ))
    assert_size_stride(primals_18, (96, ), (1, ))
    assert_size_stride(primals_19, (96, ), (1, ))
    assert_size_stride(primals_20, (288, 96), (96, 1))
    assert_size_stride(primals_21, (288, ), (1, ))
    assert_size_stride(primals_22, (96, 96), (96, 1))
    assert_size_stride(primals_23, (96, ), (1, ))
    assert_size_stride(primals_24, (96, ), (1, ))
    assert_size_stride(primals_25, (96, ), (1, ))
    assert_size_stride(primals_26, (384, 96), (96, 1))
    assert_size_stride(primals_27, (384, ), (1, ))
    assert_size_stride(primals_28, (96, 384), (384, 1))
    assert_size_stride(primals_29, (96, ), (1, ))
    assert_size_stride(primals_30, (96, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_32, (192, 96), (96, 1))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (576, 96), (96, 1))
    assert_size_stride(primals_35, (576, ), (1, ))
    assert_size_stride(primals_36, (192, 192), (192, 1))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (768, 192), (192, 1))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (192, 768), (768, 1))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_44, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (576, 192), (192, 1))
    assert_size_stride(primals_47, (576, ), (1, ))
    assert_size_stride(primals_48, (192, 192), (192, 1))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (768, 192), (192, 1))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (192, 768), (768, 1))
    assert_size_stride(primals_55, (192, ), (1, ))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_57, (192, ), (1, ))
    assert_size_stride(primals_58, (576, 192), (192, 1))
    assert_size_stride(primals_59, (576, ), (1, ))
    assert_size_stride(primals_60, (192, 192), (192, 1))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (192, ), (1, ))
    assert_size_stride(primals_63, (192, ), (1, ))
    assert_size_stride(primals_64, (768, 192), (192, 1))
    assert_size_stride(primals_65, (768, ), (1, ))
    assert_size_stride(primals_66, (192, 768), (768, 1))
    assert_size_stride(primals_67, (192, ), (1, ))
    assert_size_stride(primals_68, (192, ), (1, ))
    assert_size_stride(primals_69, (192, ), (1, ))
    assert_size_stride(primals_70, (384, 192), (192, 1))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_72, (1152, 192), (192, 1))
    assert_size_stride(primals_73, (1152, ), (1, ))
    assert_size_stride(primals_74, (384, 384), (384, 1))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, ), (1, ))
    assert_size_stride(primals_77, (384, ), (1, ))
    assert_size_stride(primals_78, (1536, 384), (384, 1))
    assert_size_stride(primals_79, (1536, ), (1, ))
    assert_size_stride(primals_80, (384, 1536), (1536, 1))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_82, (384, ), (1, ))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (1152, 384), (384, 1))
    assert_size_stride(primals_85, (1152, ), (1, ))
    assert_size_stride(primals_86, (384, 384), (384, 1))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, ), (1, ))
    assert_size_stride(primals_89, (384, ), (1, ))
    assert_size_stride(primals_90, (1536, 384), (384, 1))
    assert_size_stride(primals_91, (1536, ), (1, ))
    assert_size_stride(primals_92, (384, 1536), (1536, 1))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (1152, 384), (384, 1))
    assert_size_stride(primals_97, (1152, ), (1, ))
    assert_size_stride(primals_98, (384, 384), (384, 1))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_102, (1536, 384), (384, 1))
    assert_size_stride(primals_103, (1536, ), (1, ))
    assert_size_stride(primals_104, (384, 1536), (1536, 1))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (384, ), (1, ))
    assert_size_stride(primals_108, (1152, 384), (384, 1))
    assert_size_stride(primals_109, (1152, ), (1, ))
    assert_size_stride(primals_110, (384, 384), (384, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, ), (1, ))
    assert_size_stride(primals_113, (384, ), (1, ))
    assert_size_stride(primals_114, (1536, 384), (384, 1))
    assert_size_stride(primals_115, (1536, ), (1, ))
    assert_size_stride(primals_116, (384, 1536), (1536, 1))
    assert_size_stride(primals_117, (384, ), (1, ))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (384, ), (1, ))
    assert_size_stride(primals_120, (1152, 384), (384, 1))
    assert_size_stride(primals_121, (1152, ), (1, ))
    assert_size_stride(primals_122, (384, 384), (384, 1))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, ), (1, ))
    assert_size_stride(primals_125, (384, ), (1, ))
    assert_size_stride(primals_126, (1536, 384), (384, 1))
    assert_size_stride(primals_127, (1536, ), (1, ))
    assert_size_stride(primals_128, (384, 1536), (1536, 1))
    assert_size_stride(primals_129, (384, ), (1, ))
    assert_size_stride(primals_130, (384, ), (1, ))
    assert_size_stride(primals_131, (384, ), (1, ))
    assert_size_stride(primals_132, (1152, 384), (384, 1))
    assert_size_stride(primals_133, (1152, ), (1, ))
    assert_size_stride(primals_134, (384, 384), (384, 1))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (384, ), (1, ))
    assert_size_stride(primals_138, (1536, 384), (384, 1))
    assert_size_stride(primals_139, (1536, ), (1, ))
    assert_size_stride(primals_140, (384, 1536), (1536, 1))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (384, ), (1, ))
    assert_size_stride(primals_144, (1152, 384), (384, 1))
    assert_size_stride(primals_145, (1152, ), (1, ))
    assert_size_stride(primals_146, (384, 384), (384, 1))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (1536, 384), (384, 1))
    assert_size_stride(primals_151, (1536, ), (1, ))
    assert_size_stride(primals_152, (384, 1536), (1536, 1))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (384, ), (1, ))
    assert_size_stride(primals_155, (384, ), (1, ))
    assert_size_stride(primals_156, (1152, 384), (384, 1))
    assert_size_stride(primals_157, (1152, ), (1, ))
    assert_size_stride(primals_158, (384, 384), (384, 1))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, ), (1, ))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (1536, 384), (384, 1))
    assert_size_stride(primals_163, (1536, ), (1, ))
    assert_size_stride(primals_164, (384, 1536), (1536, 1))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (384, ), (1, ))
    assert_size_stride(primals_168, (1152, 384), (384, 1))
    assert_size_stride(primals_169, (1152, ), (1, ))
    assert_size_stride(primals_170, (384, 384), (384, 1))
    assert_size_stride(primals_171, (384, ), (1, ))
    assert_size_stride(primals_172, (384, ), (1, ))
    assert_size_stride(primals_173, (384, ), (1, ))
    assert_size_stride(primals_174, (1536, 384), (384, 1))
    assert_size_stride(primals_175, (1536, ), (1, ))
    assert_size_stride(primals_176, (384, 1536), (1536, 1))
    assert_size_stride(primals_177, (384, ), (1, ))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (1152, 384), (384, 1))
    assert_size_stride(primals_181, (1152, ), (1, ))
    assert_size_stride(primals_182, (384, 384), (384, 1))
    assert_size_stride(primals_183, (384, ), (1, ))
    assert_size_stride(primals_184, (384, ), (1, ))
    assert_size_stride(primals_185, (384, ), (1, ))
    assert_size_stride(primals_186, (1536, 384), (384, 1))
    assert_size_stride(primals_187, (1536, ), (1, ))
    assert_size_stride(primals_188, (384, 1536), (1536, 1))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (384, ), (1, ))
    assert_size_stride(primals_191, (384, ), (1, ))
    assert_size_stride(primals_192, (1152, 384), (384, 1))
    assert_size_stride(primals_193, (1152, ), (1, ))
    assert_size_stride(primals_194, (384, 384), (384, 1))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (1536, 384), (384, 1))
    assert_size_stride(primals_199, (1536, ), (1, ))
    assert_size_stride(primals_200, (384, 1536), (1536, 1))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_203, (384, ), (1, ))
    assert_size_stride(primals_204, (1152, 384), (384, 1))
    assert_size_stride(primals_205, (1152, ), (1, ))
    assert_size_stride(primals_206, (384, 384), (384, 1))
    assert_size_stride(primals_207, (384, ), (1, ))
    assert_size_stride(primals_208, (384, ), (1, ))
    assert_size_stride(primals_209, (384, ), (1, ))
    assert_size_stride(primals_210, (1536, 384), (384, 1))
    assert_size_stride(primals_211, (1536, ), (1, ))
    assert_size_stride(primals_212, (384, 1536), (1536, 1))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_214, (384, ), (1, ))
    assert_size_stride(primals_215, (384, ), (1, ))
    assert_size_stride(primals_216, (1152, 384), (384, 1))
    assert_size_stride(primals_217, (1152, ), (1, ))
    assert_size_stride(primals_218, (384, 384), (384, 1))
    assert_size_stride(primals_219, (384, ), (1, ))
    assert_size_stride(primals_220, (384, ), (1, ))
    assert_size_stride(primals_221, (384, ), (1, ))
    assert_size_stride(primals_222, (1536, 384), (384, 1))
    assert_size_stride(primals_223, (1536, ), (1, ))
    assert_size_stride(primals_224, (384, 1536), (1536, 1))
    assert_size_stride(primals_225, (384, ), (1, ))
    assert_size_stride(primals_226, (384, ), (1, ))
    assert_size_stride(primals_227, (384, ), (1, ))
    assert_size_stride(primals_228, (1152, 384), (384, 1))
    assert_size_stride(primals_229, (1152, ), (1, ))
    assert_size_stride(primals_230, (384, 384), (384, 1))
    assert_size_stride(primals_231, (384, ), (1, ))
    assert_size_stride(primals_232, (384, ), (1, ))
    assert_size_stride(primals_233, (384, ), (1, ))
    assert_size_stride(primals_234, (1536, 384), (384, 1))
    assert_size_stride(primals_235, (1536, ), (1, ))
    assert_size_stride(primals_236, (384, 1536), (1536, 1))
    assert_size_stride(primals_237, (384, ), (1, ))
    assert_size_stride(primals_238, (384, ), (1, ))
    assert_size_stride(primals_239, (384, ), (1, ))
    assert_size_stride(primals_240, (1152, 384), (384, 1))
    assert_size_stride(primals_241, (1152, ), (1, ))
    assert_size_stride(primals_242, (384, 384), (384, 1))
    assert_size_stride(primals_243, (384, ), (1, ))
    assert_size_stride(primals_244, (384, ), (1, ))
    assert_size_stride(primals_245, (384, ), (1, ))
    assert_size_stride(primals_246, (1536, 384), (384, 1))
    assert_size_stride(primals_247, (1536, ), (1, ))
    assert_size_stride(primals_248, (384, 1536), (1536, 1))
    assert_size_stride(primals_249, (384, ), (1, ))
    assert_size_stride(primals_250, (384, ), (1, ))
    assert_size_stride(primals_251, (384, ), (1, ))
    assert_size_stride(primals_252, (1152, 384), (384, 1))
    assert_size_stride(primals_253, (1152, ), (1, ))
    assert_size_stride(primals_254, (384, 384), (384, 1))
    assert_size_stride(primals_255, (384, ), (1, ))
    assert_size_stride(primals_256, (384, ), (1, ))
    assert_size_stride(primals_257, (384, ), (1, ))
    assert_size_stride(primals_258, (1536, 384), (384, 1))
    assert_size_stride(primals_259, (1536, ), (1, ))
    assert_size_stride(primals_260, (384, 1536), (1536, 1))
    assert_size_stride(primals_261, (384, ), (1, ))
    assert_size_stride(primals_262, (384, ), (1, ))
    assert_size_stride(primals_263, (384, ), (1, ))
    assert_size_stride(primals_264, (768, 384), (384, 1))
    assert_size_stride(primals_265, (768, ), (1, ))
    assert_size_stride(primals_266, (2304, 384), (384, 1))
    assert_size_stride(primals_267, (2304, ), (1, ))
    assert_size_stride(primals_268, (768, 768), (768, 1))
    assert_size_stride(primals_269, (768, ), (1, ))
    assert_size_stride(primals_270, (768, ), (1, ))
    assert_size_stride(primals_271, (768, ), (1, ))
    assert_size_stride(primals_272, (3072, 768), (768, 1))
    assert_size_stride(primals_273, (3072, ), (1, ))
    assert_size_stride(primals_274, (768, 3072), (3072, 1))
    assert_size_stride(primals_275, (768, ), (1, ))
    assert_size_stride(primals_276, (768, ), (1, ))
    assert_size_stride(primals_277, (768, ), (1, ))
    assert_size_stride(primals_278, (2304, 768), (768, 1))
    assert_size_stride(primals_279, (2304, ), (1, ))
    assert_size_stride(primals_280, (768, 768), (768, 1))
    assert_size_stride(primals_281, (768, ), (1, ))
    assert_size_stride(primals_282, (768, ), (1, ))
    assert_size_stride(primals_283, (768, ), (1, ))
    assert_size_stride(primals_284, (3072, 768), (768, 1))
    assert_size_stride(primals_285, (3072, ), (1, ))
    assert_size_stride(primals_286, (768, 3072), (3072, 1))
    assert_size_stride(primals_287, (768, ), (1, ))
    assert_size_stride(primals_288, (768, ), (1, ))
    assert_size_stride(primals_289, (768, ), (1, ))
    assert_size_stride(primals_290, (2304, 768), (768, 1))
    assert_size_stride(primals_291, (2304, ), (1, ))
    assert_size_stride(primals_292, (768, 768), (768, 1))
    assert_size_stride(primals_293, (768, ), (1, ))
    assert_size_stride(primals_294, (768, ), (1, ))
    assert_size_stride(primals_295, (768, ), (1, ))
    assert_size_stride(primals_296, (3072, 768), (768, 1))
    assert_size_stride(primals_297, (3072, ), (1, ))
    assert_size_stride(primals_298, (768, 3072), (3072, 1))
    assert_size_stride(primals_299, (768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(4, 4), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 96, 16, 16), (24576, 256, 16, 1))
        buf1 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.floor, aten._to_copy, aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_0.run(buf1, 16, grid=grid(16), stream=stream0)
        buf2 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clamp_floor_sub_0.run(buf2, 16, grid=grid(16), stream=stream0)
        buf3 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_1.run(buf3, 16, grid=grid(16), stream=stream0)
        buf4 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_2.run(buf4, 16, grid=grid(16), stream=stream0)
        buf5 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3.run(buf5, 16, grid=grid(16), stream=stream0)
        buf11 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_2.run(buf11, 16, grid=grid(16), stream=stream0)
        buf14 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.floor, aten._to_copy, aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_3.run(buf14, 16, grid=grid(16), stream=stream0)
        buf8 = empty_strided_cuda((16, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.floor, aten._to_copy, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_clamp_floor_mul_sub_1.run(buf8, 16, grid=grid(16), stream=stream0)
        buf6 = empty_strided_cuda((1, 96, 16, 16), (24576, 256, 16, 1), torch.float32)
        buf7 = buf6; del buf6  # reuse
        buf17 = buf7; del buf7  # reuse
        buf18 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [pos_embed], Original ATen: [aten.arange, aten._to_copy, aten.add, aten.mul, aten.sub, aten.floor, aten.clamp, aten.rsub, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy__unsafe_index_add_arange_clamp_floor_mul_rsub_sub_4.run(buf18, buf1, buf2, primals_5, buf3, buf4, buf5, buf8, buf11, buf14, 24576, grid=grid(24576), stream=stream0)
        del primals_5
        buf19 = empty_strided_cuda((4, 16, 16, 1), (256, 16, 1, 1024), torch.float32)
        buf20 = empty_strided_cuda((4, 16, 16, 1), (256, 16, 1, 1024), torch.float32)
        buf590 = empty_strided_cuda((4, 16, 16, 1), (256, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_5.run(buf0, primals_2, buf18, primals_4, buf19, buf20, buf590, 1024, 96, grid=grid(1024), stream=stream0)
        buf22 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, x_3], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_6.run(buf0, primals_2, buf18, primals_4, buf19, buf20, buf22, 384, 256, grid=grid(384, 256), stream=stream0)
        buf23 = empty_strided_cuda((4, 2, 2, 8, 8, 96), (24576, 12288, 6144, 768, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf22, primals_6, primals_7, buf23, 98304, grid=grid(98304), stream=stream0)
        del primals_7
        buf24 = empty_strided_cuda((1024, 288), (288, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_9, reinterpret_tensor(buf23, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_8, (96, 288), (1, 96), 0), alpha=1, beta=1, out=buf24)
        del primals_9
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf25 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf24, (16, 1, 64, 96), (18432, 96, 288, 1), 0), reinterpret_tensor(buf24, (16, 1, 64, 96), (18432, 96, 288, 1), 96), reinterpret_tensor(buf24, (16, 1, 64, 96), (18432, 96, 288, 1), 192), None, True)
        buf26 = buf25[0]
        buf27 = buf25[1]
        buf28 = buf25[2]
        buf29 = buf25[3]
        del buf25
        buf30 = empty_strided_cuda((1024, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_10, (96, 96), (1, 96), 0), out=buf30)
        buf31 = reinterpret_tensor(buf0, (4, 16, 16, 96), (24576, 16, 1, 256), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_2, x_11], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_8.run(buf31, primals_2, buf18, primals_4, buf30, primals_11, 384, 256, grid=grid(384, 256), stream=stream0)
        del primals_11
        del primals_2
        del primals_4
        buf35 = reinterpret_tensor(buf30, (4, 16, 16, 96), (24576, 1536, 96, 1), 0); del buf30  # reuse
        buf36 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        buf589 = reinterpret_tensor(buf20, (4, 16, 16, 1), (256, 16, 1, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_native_layer_norm_native_layer_norm_backward_9.run(buf31, primals_12, primals_13, buf35, buf36, buf589, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_13
        buf37 = empty_strided_cuda((1024, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_15, reinterpret_tensor(buf36, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_14, (96, 384), (1, 96), 0), alpha=1, beta=1, out=buf37)
        del primals_15
        buf38 = empty_strided_cuda((4, 16, 16, 384), (98304, 6144, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_10.run(buf37, buf38, 393216, grid=grid(393216), stream=stream0)
        buf39 = empty_strided_cuda((1024, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf38, (1024, 384), (384, 1), 0), reinterpret_tensor(primals_16, (384, 96), (1, 384), 0), out=buf39)
        buf43 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        buf588 = reinterpret_tensor(buf19, (4, 16, 16, 1), (256, 16, 1, 1), 0); del buf19  # reuse
        # Topologically Sorted Source Nodes: [x_14, x_15], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf31, buf39, primals_17, buf43, buf588, 1024, 96, grid=grid(1024), stream=stream0)
        buf44 = empty_strided_cuda((4, 2, 2, 8, 8, 96), (24576, 12288, 6144, 768, 96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf43, primals_18, primals_19, buf44, 98304, grid=grid(98304), stream=stream0)
        del primals_19
        buf45 = empty_strided_cuda((1024, 288), (288, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, reinterpret_tensor(buf44, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_20, (96, 288), (1, 96), 0), alpha=1, beta=1, out=buf45)
        del primals_21
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf46 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf45, (16, 1, 64, 96), (18432, 96, 288, 1), 0), reinterpret_tensor(buf45, (16, 1, 64, 96), (18432, 96, 288, 1), 96), reinterpret_tensor(buf45, (16, 1, 64, 96), (18432, 96, 288, 1), 192), None, True)
        buf47 = buf46[0]
        buf48 = buf46[1]
        buf49 = buf46[2]
        buf50 = buf46[3]
        del buf46
        buf51 = empty_strided_cuda((1024, 96), (96, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf47, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_22, (96, 96), (1, 96), 0), out=buf51)
        buf52 = reinterpret_tensor(buf39, (4, 16, 16, 96), (24576, 1536, 96, 1), 0); del buf39  # reuse
        buf56 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        buf57 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        buf587 = empty_strided_cuda((4, 16, 16, 1), (256, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14, x_23, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf52, buf31, primals_17, buf51, primals_23, primals_24, primals_25, buf56, buf57, buf587, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_17
        del primals_23
        del primals_25
        buf58 = empty_strided_cuda((1024, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, reinterpret_tensor(buf57, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_26, (96, 384), (1, 96), 0), alpha=1, beta=1, out=buf58)
        del primals_27
        buf59 = empty_strided_cuda((4, 16, 16, 384), (98304, 6144, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_10.run(buf58, buf59, 393216, grid=grid(393216), stream=stream0)
        buf60 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf59, (1024, 384), (384, 1), 0), reinterpret_tensor(primals_28, (384, 96), (1, 384), 0), out=buf60)
        buf61 = buf52; del buf52  # reuse
        buf66 = reinterpret_tensor(buf31, (4, 16, 16, 96), (24576, 1536, 96, 1), 0); del buf31  # reuse
        buf67 = empty_strided_cuda((4, 16, 16, 96), (24576, 1536, 96, 1), torch.float32)
        buf586 = empty_strided_cuda((4, 16, 16, 1), (256, 16, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_26, x_27], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13.run(buf61, buf60, primals_29, primals_30, primals_31, buf66, buf67, buf586, 1024, 96, grid=grid(1024), stream=stream0)
        del primals_29
        del primals_31
        buf62 = reinterpret_tensor(buf60, (4, 96, 16, 16), (24576, 256, 16, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_26, feats], Original ATen: [aten.add, aten.permute]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_permute_14.run(buf61, buf62, 384, 256, grid=grid(384, 256), stream=stream0)
        buf68 = empty_strided_cuda((1024, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, reinterpret_tensor(buf67, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_32, (96, 192), (1, 96), 0), alpha=1, beta=1, out=buf68)
        del primals_33
        buf70 = reinterpret_tensor(buf61, (4, 2, 2, 8, 8, 96), (24576, 12288, 6144, 768, 96, 1), 0); del buf61  # reuse
        # Topologically Sorted Source Nodes: [contiguous_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_15.run(buf67, buf70, 98304, grid=grid(98304), stream=stream0)
        buf71 = empty_strided_cuda((1024, 576), (576, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, reinterpret_tensor(buf70, (1024, 96), (96, 1), 0), reinterpret_tensor(primals_34, (96, 576), (1, 96), 0), alpha=1, beta=1, out=buf71)
        del primals_35
        buf72 = empty_strided_cuda((16, 192, 4, 4), (3072, 1, 768, 192), torch.int8)
        buf73 = empty_strided_cuda((16, 192, 4, 4), (3072, 1, 768, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_16.run(buf71, buf72, buf73, 49152, grid=grid(49152), stream=stream0)
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf74 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf73, (16, 2, 16, 96), (3072, 96, 192, 1), 0), reinterpret_tensor(buf71, (16, 2, 64, 96), (36864, 96, 576, 1), 192), reinterpret_tensor(buf71, (16, 2, 64, 96), (36864, 96, 576, 1), 384), None, True)
        buf75 = buf74[0]
        buf79 = empty_strided_cuda((256, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf75, (256, 192), (192, 1), 0), reinterpret_tensor(primals_36, (192, 192), (1, 192), 0), out=buf79)
        buf69 = empty_strided_cuda((4, 192, 8, 8), (12288, 1, 1536, 192), torch.int8)
        buf80 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf84 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf85 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf585 = empty_strided_cuda((4, 8, 8, 1), (64, 8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_29, x_41, layer_norm_5], Original ATen: [aten.max_pool2d_with_indices, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_17.run(buf68, buf79, primals_37, primals_38, primals_39, buf69, buf80, buf84, buf85, buf585, 256, 192, grid=grid(256), stream=stream0)
        del primals_37
        del primals_39
        buf76 = buf74[1]
        buf77 = buf74[2]
        buf78 = buf74[3]
        del buf74
        buf86 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_41, reinterpret_tensor(buf85, (256, 192), (192, 1), 0), reinterpret_tensor(primals_40, (192, 768), (1, 192), 0), alpha=1, beta=1, out=buf86)
        del primals_41
        buf87 = empty_strided_cuda((4, 8, 8, 768), (49152, 6144, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_18.run(buf86, buf87, 196608, grid=grid(196608), stream=stream0)
        buf88 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf87, (256, 768), (768, 1), 0), reinterpret_tensor(primals_42, (768, 192), (1, 768), 0), out=buf88)
        buf92 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf584 = empty_strided_cuda((4, 8, 8, 1), (64, 8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_44, x_45], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf80, buf88, primals_43, buf92, buf584, 256, 192, grid=grid(256), stream=stream0)
        buf93 = empty_strided_cuda((4, 2, 2, 4, 4, 192), (12288, 6144, 3072, 768, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_20.run(buf92, primals_44, primals_45, buf93, 49152, grid=grid(49152), stream=stream0)
        del primals_45
        buf94 = empty_strided_cuda((256, 576), (576, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_47, reinterpret_tensor(buf93, (256, 192), (192, 1), 0), reinterpret_tensor(primals_46, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf94)
        del primals_47
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf95 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf94, (16, 2, 16, 96), (9216, 96, 576, 1), 0), reinterpret_tensor(buf94, (16, 2, 16, 96), (9216, 96, 576, 1), 192), reinterpret_tensor(buf94, (16, 2, 16, 96), (9216, 96, 576, 1), 384), None, True)
        buf96 = buf95[0]
        buf97 = buf95[1]
        buf98 = buf95[2]
        buf99 = buf95[3]
        del buf95
        buf100 = empty_strided_cuda((256, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf96, (256, 192), (192, 1), 0), reinterpret_tensor(primals_48, (192, 192), (1, 192), 0), out=buf100)
        buf101 = buf80; del buf80  # reuse
        buf105 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf106 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf583 = empty_strided_cuda((4, 8, 8, 1), (64, 8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_44, x_53, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf101, buf88, primals_43, buf100, primals_49, primals_50, primals_51, buf105, buf106, buf583, 256, 192, grid=grid(256), stream=stream0)
        del primals_43
        del primals_49
        del primals_51
        buf107 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_53, reinterpret_tensor(buf106, (256, 192), (192, 1), 0), reinterpret_tensor(primals_52, (192, 768), (1, 192), 0), alpha=1, beta=1, out=buf107)
        del primals_53
        buf108 = empty_strided_cuda((4, 8, 8, 768), (49152, 6144, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_18.run(buf107, buf108, 196608, grid=grid(196608), stream=stream0)
        buf109 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf108, (256, 768), (768, 1), 0), reinterpret_tensor(primals_54, (768, 192), (1, 768), 0), out=buf109)
        buf113 = reinterpret_tensor(buf100, (4, 8, 8, 192), (12288, 1536, 192, 1), 0); del buf100  # reuse
        buf582 = empty_strided_cuda((4, 8, 8, 1), (64, 8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_56, x_57], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_19.run(buf101, buf109, primals_55, buf113, buf582, 256, 192, grid=grid(256), stream=stream0)
        buf114 = empty_strided_cuda((4, 2, 2, 4, 4, 192), (12288, 6144, 3072, 768, 192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_20.run(buf113, primals_56, primals_57, buf114, 49152, grid=grid(49152), stream=stream0)
        del primals_57
        buf115 = empty_strided_cuda((256, 576), (576, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_59, reinterpret_tensor(buf114, (256, 192), (192, 1), 0), reinterpret_tensor(primals_58, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf115)
        del primals_59
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf116 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf115, (16, 2, 16, 96), (9216, 96, 576, 1), 0), reinterpret_tensor(buf115, (16, 2, 16, 96), (9216, 96, 576, 1), 192), reinterpret_tensor(buf115, (16, 2, 16, 96), (9216, 96, 576, 1), 384), None, True)
        buf117 = buf116[0]
        buf118 = buf116[1]
        buf119 = buf116[2]
        buf120 = buf116[3]
        del buf116
        buf121 = empty_strided_cuda((256, 192), (192, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf117, (256, 192), (192, 1), 0), reinterpret_tensor(primals_60, (192, 192), (1, 192), 0), out=buf121)
        buf122 = buf101; del buf101  # reuse
        buf126 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf127 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf581 = empty_strided_cuda((4, 8, 8, 1), (64, 8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_56, x_65, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_21.run(buf122, buf109, primals_55, buf121, primals_61, primals_62, primals_63, buf126, buf127, buf581, 256, 192, grid=grid(256), stream=stream0)
        del primals_55
        del primals_61
        del primals_63
        buf128 = empty_strided_cuda((256, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, reinterpret_tensor(buf127, (256, 192), (192, 1), 0), reinterpret_tensor(primals_64, (192, 768), (1, 192), 0), alpha=1, beta=1, out=buf128)
        del primals_65
        buf129 = empty_strided_cuda((4, 8, 8, 768), (49152, 6144, 768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_18.run(buf128, buf129, 196608, grid=grid(196608), stream=stream0)
        buf130 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf129, (256, 768), (768, 1), 0), reinterpret_tensor(primals_66, (768, 192), (1, 768), 0), out=buf130)
        buf131 = buf122; del buf122  # reuse
        buf135 = reinterpret_tensor(buf109, (4, 8, 8, 192), (12288, 1536, 192, 1), 0); del buf109  # reuse
        buf136 = empty_strided_cuda((4, 8, 8, 192), (12288, 1536, 192, 1), torch.float32)
        buf580 = empty_strided_cuda((4, 8, 8, 1), (64, 8, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_68, x_69], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_22.run(buf131, buf130, primals_67, primals_68, primals_69, buf135, buf136, buf580, 256, 192, grid=grid(256), stream=stream0)
        del primals_67
        del primals_69
        buf137 = empty_strided_cuda((256, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_71, reinterpret_tensor(buf136, (256, 192), (192, 1), 0), reinterpret_tensor(primals_70, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf137)
        del primals_71
        buf139 = reinterpret_tensor(buf130, (4, 2, 2, 4, 4, 192), (12288, 6144, 3072, 768, 192, 1), 0); del buf130  # reuse
        # Topologically Sorted Source Nodes: [contiguous_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_23.run(buf136, buf139, 49152, grid=grid(49152), stream=stream0)
        buf140 = empty_strided_cuda((256, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_73, reinterpret_tensor(buf139, (256, 192), (192, 1), 0), reinterpret_tensor(primals_72, (192, 1152), (1, 192), 0), alpha=1, beta=1, out=buf140)
        del primals_73
        buf141 = empty_strided_cuda((16, 384, 2, 2), (1536, 1, 768, 384), torch.int8)
        buf142 = reinterpret_tensor(buf18, (16, 384, 2, 2), (1536, 1, 768, 384), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_24.run(buf140, buf141, buf142, 24576, grid=grid(24576), stream=stream0)
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf143 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf142, (16, 4, 4, 96), (1536, 96, 384, 1), 0), reinterpret_tensor(buf140, (16, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf140, (16, 4, 16, 96), (18432, 96, 1152, 1), 768), None, True)
        buf144 = buf143[0]
        buf148 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf144, (64, 384), (384, 1), 0), reinterpret_tensor(primals_74, (384, 384), (1, 384), 0), out=buf148)
        buf138 = empty_strided_cuda((4, 384, 4, 4), (6144, 1, 1536, 384), torch.int8)
        buf149 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf153 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf154 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf579 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_71, x_83, layer_norm_11], Original ATen: [aten.max_pool2d_with_indices, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_25.run(buf137, buf148, primals_75, primals_76, primals_77, buf138, buf149, buf153, buf154, buf579, 64, 384, grid=grid(64), stream=stream0)
        del primals_75
        del primals_77
        buf145 = buf143[1]
        buf146 = buf143[2]
        buf147 = buf143[3]
        del buf143
        buf155 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_79, reinterpret_tensor(buf154, (64, 384), (384, 1), 0), reinterpret_tensor(primals_78, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf155)
        del primals_79
        buf156 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf155, buf156, 98304, grid=grid(98304), stream=stream0)
        buf157 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_85], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf156, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_80, (1536, 384), (1, 1536), 0), out=buf157)
        buf161 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf578 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_86, x_87], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf149, buf157, primals_81, buf161, buf578, 64, 384, grid=grid(64), stream=stream0)
        buf162 = empty_strided_cuda((4, 14, 14, 384), (75264, 5376, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_87, x_88], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf161, primals_82, primals_83, buf162, 301056, grid=grid(301056), stream=stream0)
        del primals_83
        buf163 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_85, reinterpret_tensor(buf162, (784, 384), (384, 1), 0), reinterpret_tensor(primals_84, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf163)
        del primals_85
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf164 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf163, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf163, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf163, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf165 = buf164[0]
        buf166 = buf164[1]
        buf167 = buf164[2]
        buf168 = buf164[3]
        del buf164
        buf169 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf165, (784, 384), (384, 1), 0), reinterpret_tensor(primals_86, (384, 384), (1, 384), 0), out=buf169)
        buf170 = buf149; del buf149  # reuse
        buf174 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf175 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf577 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_86, x_96, x_97, layer_norm_13], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf170, buf157, primals_81, buf169, primals_87, primals_88, primals_89, buf174, buf175, buf577, 64, 384, grid=grid(64), stream=stream0)
        del primals_81
        del primals_87
        del primals_89
        buf176 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_91, reinterpret_tensor(buf175, (64, 384), (384, 1), 0), reinterpret_tensor(primals_90, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf176)
        del primals_91
        buf177 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf176, buf177, 98304, grid=grid(98304), stream=stream0)
        buf178 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf177, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_92, (1536, 384), (1, 1536), 0), out=buf178)
        buf182 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf576 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_100, x_101], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf170, buf178, primals_93, buf182, buf576, 64, 384, grid=grid(64), stream=stream0)
        buf183 = reinterpret_tensor(buf169, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_101, x_102], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf182, primals_94, primals_95, buf183, 301056, grid=grid(301056), stream=stream0)
        del primals_95
        buf184 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_97, reinterpret_tensor(buf183, (784, 384), (384, 1), 0), reinterpret_tensor(primals_96, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf184)
        del primals_97
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf185 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf184, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf184, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf184, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf186 = buf185[0]
        buf187 = buf185[1]
        buf188 = buf185[2]
        buf189 = buf185[3]
        del buf185
        buf190 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf186, (784, 384), (384, 1), 0), reinterpret_tensor(primals_98, (384, 384), (1, 384), 0), out=buf190)
        buf191 = buf170; del buf170  # reuse
        buf195 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf196 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf575 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_100, x_110, x_111, layer_norm_15], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf191, buf178, primals_93, buf190, primals_99, primals_100, primals_101, buf195, buf196, buf575, 64, 384, grid=grid(64), stream=stream0)
        del primals_101
        del primals_93
        del primals_99
        buf197 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_103, reinterpret_tensor(buf196, (64, 384), (384, 1), 0), reinterpret_tensor(primals_102, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf197)
        del primals_103
        buf198 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_112], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf197, buf198, 98304, grid=grid(98304), stream=stream0)
        buf199 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_113], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf198, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_104, (1536, 384), (1, 1536), 0), out=buf199)
        buf203 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf574 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_114, x_115], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf191, buf199, primals_105, buf203, buf574, 64, 384, grid=grid(64), stream=stream0)
        buf204 = reinterpret_tensor(buf190, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf190  # reuse
        # Topologically Sorted Source Nodes: [x_115, x_116], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf203, primals_106, primals_107, buf204, 301056, grid=grid(301056), stream=stream0)
        del primals_107
        buf205 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_109, reinterpret_tensor(buf204, (784, 384), (384, 1), 0), reinterpret_tensor(primals_108, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf205)
        del primals_109
        # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf206 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf205, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf205, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf205, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf207 = buf206[0]
        buf208 = buf206[1]
        buf209 = buf206[2]
        buf210 = buf206[3]
        del buf206
        buf211 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf207, (784, 384), (384, 1), 0), reinterpret_tensor(primals_110, (384, 384), (1, 384), 0), out=buf211)
        buf212 = buf191; del buf191  # reuse
        buf216 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf217 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf573 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_114, x_124, x_125, layer_norm_17], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf212, buf199, primals_105, buf211, primals_111, primals_112, primals_113, buf216, buf217, buf573, 64, 384, grid=grid(64), stream=stream0)
        del primals_105
        del primals_111
        del primals_113
        buf218 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_115, reinterpret_tensor(buf217, (64, 384), (384, 1), 0), reinterpret_tensor(primals_114, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf218)
        del primals_115
        buf219 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf218, buf219, 98304, grid=grid(98304), stream=stream0)
        buf220 = buf199; del buf199  # reuse
        # Topologically Sorted Source Nodes: [x_127], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf219, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_116, (1536, 384), (1, 1536), 0), out=buf220)
        buf224 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf572 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_128, x_129], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf212, buf220, primals_117, buf224, buf572, 64, 384, grid=grid(64), stream=stream0)
        buf225 = reinterpret_tensor(buf211, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [x_129, x_130], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf224, primals_118, primals_119, buf225, 301056, grid=grid(301056), stream=stream0)
        del primals_119
        buf226 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_121, reinterpret_tensor(buf225, (784, 384), (384, 1), 0), reinterpret_tensor(primals_120, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf226)
        del primals_121
        # Topologically Sorted Source Nodes: [x_132], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf227 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf226, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf226, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf226, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf228 = buf227[0]
        buf229 = buf227[1]
        buf230 = buf227[2]
        buf231 = buf227[3]
        del buf227
        buf232 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf228, (784, 384), (384, 1), 0), reinterpret_tensor(primals_122, (384, 384), (1, 384), 0), out=buf232)
        buf233 = buf212; del buf212  # reuse
        buf237 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf238 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf571 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_128, x_138, x_139, layer_norm_19], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf233, buf220, primals_117, buf232, primals_123, primals_124, primals_125, buf237, buf238, buf571, 64, 384, grid=grid(64), stream=stream0)
        del primals_117
        del primals_123
        del primals_125
        buf239 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_127, reinterpret_tensor(buf238, (64, 384), (384, 1), 0), reinterpret_tensor(primals_126, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf239)
        del primals_127
        buf240 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf239, buf240, 98304, grid=grid(98304), stream=stream0)
        buf241 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_141], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf240, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_128, (1536, 384), (1, 1536), 0), out=buf241)
        buf245 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf570 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_142, x_143], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf233, buf241, primals_129, buf245, buf570, 64, 384, grid=grid(64), stream=stream0)
        buf246 = reinterpret_tensor(buf232, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf232  # reuse
        # Topologically Sorted Source Nodes: [x_143, x_144], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf245, primals_130, primals_131, buf246, 301056, grid=grid(301056), stream=stream0)
        del primals_131
        buf247 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_133, reinterpret_tensor(buf246, (784, 384), (384, 1), 0), reinterpret_tensor(primals_132, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf247)
        del primals_133
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf248 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf247, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf247, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf247, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf249 = buf248[0]
        buf250 = buf248[1]
        buf251 = buf248[2]
        buf252 = buf248[3]
        del buf248
        buf253 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf249, (784, 384), (384, 1), 0), reinterpret_tensor(primals_134, (384, 384), (1, 384), 0), out=buf253)
        buf254 = buf233; del buf233  # reuse
        buf258 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf259 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf569 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_142, x_152, x_153, layer_norm_21], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf254, buf241, primals_129, buf253, primals_135, primals_136, primals_137, buf258, buf259, buf569, 64, 384, grid=grid(64), stream=stream0)
        del primals_129
        del primals_135
        del primals_137
        buf260 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_139, reinterpret_tensor(buf259, (64, 384), (384, 1), 0), reinterpret_tensor(primals_138, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf260)
        del primals_139
        buf261 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf260, buf261, 98304, grid=grid(98304), stream=stream0)
        buf262 = buf241; del buf241  # reuse
        # Topologically Sorted Source Nodes: [x_155], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf261, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_140, (1536, 384), (1, 1536), 0), out=buf262)
        buf266 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf568 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_156, x_157], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf254, buf262, primals_141, buf266, buf568, 64, 384, grid=grid(64), stream=stream0)
        buf267 = reinterpret_tensor(buf253, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf253  # reuse
        # Topologically Sorted Source Nodes: [x_157, x_158], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf266, primals_142, primals_143, buf267, 301056, grid=grid(301056), stream=stream0)
        del primals_143
        buf268 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_145, reinterpret_tensor(buf267, (784, 384), (384, 1), 0), reinterpret_tensor(primals_144, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf268)
        del primals_145
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf269 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf268, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf268, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf268, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf270 = buf269[0]
        buf271 = buf269[1]
        buf272 = buf269[2]
        buf273 = buf269[3]
        del buf269
        buf274 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf270, (784, 384), (384, 1), 0), reinterpret_tensor(primals_146, (384, 384), (1, 384), 0), out=buf274)
        buf275 = buf254; del buf254  # reuse
        buf279 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf280 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf567 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_156, x_166, x_167, layer_norm_23], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf275, buf262, primals_141, buf274, primals_147, primals_148, primals_149, buf279, buf280, buf567, 64, 384, grid=grid(64), stream=stream0)
        del primals_141
        del primals_147
        del primals_149
        buf281 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_151, reinterpret_tensor(buf280, (64, 384), (384, 1), 0), reinterpret_tensor(primals_150, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf281)
        del primals_151
        buf282 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf281, buf282, 98304, grid=grid(98304), stream=stream0)
        buf283 = buf262; del buf262  # reuse
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf282, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_152, (1536, 384), (1, 1536), 0), out=buf283)
        buf287 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf288 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf566 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_170, x_171], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30.run(buf275, buf283, primals_153, primals_154, primals_155, buf287, buf288, buf566, 64, 384, grid=grid(64), stream=stream0)
        del primals_155
        buf289 = empty_strided_cuda((64, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_157, reinterpret_tensor(buf288, (64, 384), (384, 1), 0), reinterpret_tensor(primals_156, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf289)
        del primals_157
        # Topologically Sorted Source Nodes: [x_172], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf290 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf289, (4, 4, 16, 96), (18432, 96, 1152, 1), 0), reinterpret_tensor(buf289, (4, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf289, (4, 4, 16, 96), (18432, 96, 1152, 1), 768), None, True)
        buf291 = buf290[0]
        buf292 = buf290[1]
        buf293 = buf290[2]
        buf294 = buf290[3]
        del buf290
        buf295 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf291, (64, 384), (384, 1), 0), reinterpret_tensor(primals_158, (384, 384), (1, 384), 0), out=buf295)
        buf296 = buf275; del buf275  # reuse
        buf300 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf301 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf565 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_170, x_176, layer_norm_25], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31.run(buf296, buf283, primals_153, buf295, primals_159, primals_160, primals_161, buf300, buf301, buf565, 64, 384, grid=grid(64), stream=stream0)
        del primals_153
        del primals_159
        del primals_161
        buf302 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_163, reinterpret_tensor(buf301, (64, 384), (384, 1), 0), reinterpret_tensor(primals_162, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf302)
        del primals_163
        buf303 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_177], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf302, buf303, 98304, grid=grid(98304), stream=stream0)
        buf304 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf303, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_164, (1536, 384), (1, 1536), 0), out=buf304)
        buf308 = reinterpret_tensor(buf283, (4, 4, 4, 384), (6144, 1536, 384, 1), 0); del buf283  # reuse
        buf564 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_179, x_180], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf296, buf304, primals_165, buf308, buf564, 64, 384, grid=grid(64), stream=stream0)
        buf309 = reinterpret_tensor(buf274, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf274  # reuse
        # Topologically Sorted Source Nodes: [x_180, x_181], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf308, primals_166, primals_167, buf309, 301056, grid=grid(301056), stream=stream0)
        del primals_167
        buf310 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_169, reinterpret_tensor(buf309, (784, 384), (384, 1), 0), reinterpret_tensor(primals_168, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf310)
        del primals_169
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf311 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf310, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf310, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf310, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf312 = buf311[0]
        buf313 = buf311[1]
        buf314 = buf311[2]
        buf315 = buf311[3]
        del buf311
        buf316 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf312, (784, 384), (384, 1), 0), reinterpret_tensor(primals_170, (384, 384), (1, 384), 0), out=buf316)
        buf317 = buf296; del buf296  # reuse
        buf321 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf322 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf563 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_179, x_189, x_190, layer_norm_27], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf317, buf304, primals_165, buf316, primals_171, primals_172, primals_173, buf321, buf322, buf563, 64, 384, grid=grid(64), stream=stream0)
        del primals_165
        del primals_171
        del primals_173
        buf323 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_175, reinterpret_tensor(buf322, (64, 384), (384, 1), 0), reinterpret_tensor(primals_174, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf323)
        del primals_175
        buf324 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_191], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf323, buf324, 98304, grid=grid(98304), stream=stream0)
        buf325 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [x_192], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf324, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_176, (1536, 384), (1, 1536), 0), out=buf325)
        buf329 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf562 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_193, x_194], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf317, buf325, primals_177, buf329, buf562, 64, 384, grid=grid(64), stream=stream0)
        buf330 = reinterpret_tensor(buf316, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf316  # reuse
        # Topologically Sorted Source Nodes: [x_194, x_195], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf329, primals_178, primals_179, buf330, 301056, grid=grid(301056), stream=stream0)
        del primals_179
        buf331 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_181, reinterpret_tensor(buf330, (784, 384), (384, 1), 0), reinterpret_tensor(primals_180, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf331)
        del primals_181
        # Topologically Sorted Source Nodes: [x_197], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf332 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf331, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf331, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf331, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf333 = buf332[0]
        buf334 = buf332[1]
        buf335 = buf332[2]
        buf336 = buf332[3]
        del buf332
        buf337 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf333, (784, 384), (384, 1), 0), reinterpret_tensor(primals_182, (384, 384), (1, 384), 0), out=buf337)
        buf338 = buf317; del buf317  # reuse
        buf342 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf343 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf561 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_193, x_203, x_204, layer_norm_29], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf338, buf325, primals_177, buf337, primals_183, primals_184, primals_185, buf342, buf343, buf561, 64, 384, grid=grid(64), stream=stream0)
        del primals_177
        del primals_183
        del primals_185
        buf344 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_187, reinterpret_tensor(buf343, (64, 384), (384, 1), 0), reinterpret_tensor(primals_186, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf344)
        del primals_187
        buf345 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_205], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf344, buf345, 98304, grid=grid(98304), stream=stream0)
        buf346 = buf325; del buf325  # reuse
        # Topologically Sorted Source Nodes: [x_206], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf345, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_188, (1536, 384), (1, 1536), 0), out=buf346)
        buf350 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf560 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_207, x_208], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf338, buf346, primals_189, buf350, buf560, 64, 384, grid=grid(64), stream=stream0)
        buf351 = reinterpret_tensor(buf337, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf337  # reuse
        # Topologically Sorted Source Nodes: [x_208, x_209], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf350, primals_190, primals_191, buf351, 301056, grid=grid(301056), stream=stream0)
        del primals_191
        buf352 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_193, reinterpret_tensor(buf351, (784, 384), (384, 1), 0), reinterpret_tensor(primals_192, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf352)
        del primals_193
        # Topologically Sorted Source Nodes: [x_211], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf353 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf352, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf352, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf352, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf354 = buf353[0]
        buf355 = buf353[1]
        buf356 = buf353[2]
        buf357 = buf353[3]
        del buf353
        buf358 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_214], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf354, (784, 384), (384, 1), 0), reinterpret_tensor(primals_194, (384, 384), (1, 384), 0), out=buf358)
        buf359 = buf338; del buf338  # reuse
        buf363 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf364 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf559 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_207, x_217, x_218, layer_norm_31], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf359, buf346, primals_189, buf358, primals_195, primals_196, primals_197, buf363, buf364, buf559, 64, 384, grid=grid(64), stream=stream0)
        del primals_189
        del primals_195
        del primals_197
        buf365 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_64], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_199, reinterpret_tensor(buf364, (64, 384), (384, 1), 0), reinterpret_tensor(primals_198, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf365)
        del primals_199
        buf366 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_219], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf365, buf366, 98304, grid=grid(98304), stream=stream0)
        buf367 = buf346; del buf346  # reuse
        # Topologically Sorted Source Nodes: [x_220], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf366, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_200, (1536, 384), (1, 1536), 0), out=buf367)
        buf371 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf372 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf558 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_221, x_222], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30.run(buf359, buf367, primals_201, primals_202, primals_203, buf371, buf372, buf558, 64, 384, grid=grid(64), stream=stream0)
        del primals_203
        buf373 = empty_strided_cuda((64, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_66], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_205, reinterpret_tensor(buf372, (64, 384), (384, 1), 0), reinterpret_tensor(primals_204, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf373)
        del primals_205
        # Topologically Sorted Source Nodes: [x_223], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf374 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf373, (4, 4, 16, 96), (18432, 96, 1152, 1), 0), reinterpret_tensor(buf373, (4, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf373, (4, 4, 16, 96), (18432, 96, 1152, 1), 768), None, True)
        buf375 = buf374[0]
        buf376 = buf374[1]
        buf377 = buf374[2]
        buf378 = buf374[3]
        del buf374
        buf379 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_226], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf375, (64, 384), (384, 1), 0), reinterpret_tensor(primals_206, (384, 384), (1, 384), 0), out=buf379)
        buf380 = buf359; del buf359  # reuse
        buf384 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf385 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf557 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_221, x_227, layer_norm_33], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31.run(buf380, buf367, primals_201, buf379, primals_207, primals_208, primals_209, buf384, buf385, buf557, 64, 384, grid=grid(64), stream=stream0)
        del primals_201
        del primals_207
        del primals_209
        buf386 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_68], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_211, reinterpret_tensor(buf385, (64, 384), (384, 1), 0), reinterpret_tensor(primals_210, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf386)
        del primals_211
        buf387 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_228], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf386, buf387, 98304, grid=grid(98304), stream=stream0)
        buf388 = buf379; del buf379  # reuse
        # Topologically Sorted Source Nodes: [x_229], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf387, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_212, (1536, 384), (1, 1536), 0), out=buf388)
        buf392 = reinterpret_tensor(buf367, (4, 4, 4, 384), (6144, 1536, 384, 1), 0); del buf367  # reuse
        buf556 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_230, x_231], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf380, buf388, primals_213, buf392, buf556, 64, 384, grid=grid(64), stream=stream0)
        buf393 = reinterpret_tensor(buf358, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf358  # reuse
        # Topologically Sorted Source Nodes: [x_231, x_232], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf392, primals_214, primals_215, buf393, 301056, grid=grid(301056), stream=stream0)
        del primals_215
        buf394 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_70], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_217, reinterpret_tensor(buf393, (784, 384), (384, 1), 0), reinterpret_tensor(primals_216, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf394)
        del primals_217
        # Topologically Sorted Source Nodes: [x_234], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf395 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf394, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf394, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf394, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf396 = buf395[0]
        buf397 = buf395[1]
        buf398 = buf395[2]
        buf399 = buf395[3]
        del buf395
        buf400 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_237], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf396, (784, 384), (384, 1), 0), reinterpret_tensor(primals_218, (384, 384), (1, 384), 0), out=buf400)
        buf401 = buf380; del buf380  # reuse
        buf405 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf406 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf555 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_230, x_240, x_241, layer_norm_35], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf401, buf388, primals_213, buf400, primals_219, primals_220, primals_221, buf405, buf406, buf555, 64, 384, grid=grid(64), stream=stream0)
        del primals_213
        del primals_219
        del primals_221
        buf407 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_72], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_223, reinterpret_tensor(buf406, (64, 384), (384, 1), 0), reinterpret_tensor(primals_222, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf407)
        del primals_223
        buf408 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_242], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf407, buf408, 98304, grid=grid(98304), stream=stream0)
        buf409 = buf388; del buf388  # reuse
        # Topologically Sorted Source Nodes: [x_243], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf408, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_224, (1536, 384), (1, 1536), 0), out=buf409)
        buf413 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf554 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_244, x_245], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf401, buf409, primals_225, buf413, buf554, 64, 384, grid=grid(64), stream=stream0)
        buf414 = reinterpret_tensor(buf400, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf400  # reuse
        # Topologically Sorted Source Nodes: [x_245, x_246], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf413, primals_226, primals_227, buf414, 301056, grid=grid(301056), stream=stream0)
        del primals_227
        buf415 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_74], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_229, reinterpret_tensor(buf414, (784, 384), (384, 1), 0), reinterpret_tensor(primals_228, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf415)
        del primals_229
        # Topologically Sorted Source Nodes: [x_248], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf416 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf415, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf415, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf415, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf417 = buf416[0]
        buf418 = buf416[1]
        buf419 = buf416[2]
        buf420 = buf416[3]
        del buf416
        buf421 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_251], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf417, (784, 384), (384, 1), 0), reinterpret_tensor(primals_230, (384, 384), (1, 384), 0), out=buf421)
        buf422 = buf401; del buf401  # reuse
        buf426 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf427 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf553 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_244, x_254, x_255, layer_norm_37], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf422, buf409, primals_225, buf421, primals_231, primals_232, primals_233, buf426, buf427, buf553, 64, 384, grid=grid(64), stream=stream0)
        del primals_225
        del primals_231
        del primals_233
        buf428 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_76], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_235, reinterpret_tensor(buf427, (64, 384), (384, 1), 0), reinterpret_tensor(primals_234, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf428)
        del primals_235
        buf429 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_256], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf428, buf429, 98304, grid=grid(98304), stream=stream0)
        buf430 = buf409; del buf409  # reuse
        # Topologically Sorted Source Nodes: [x_257], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf429, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_236, (1536, 384), (1, 1536), 0), out=buf430)
        buf434 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf552 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_258, x_259], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_27.run(buf422, buf430, primals_237, buf434, buf552, 64, 384, grid=grid(64), stream=stream0)
        buf435 = reinterpret_tensor(buf421, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [x_259, x_260], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_28.run(buf434, primals_238, primals_239, buf435, 301056, grid=grid(301056), stream=stream0)
        del primals_239
        buf436 = empty_strided_cuda((784, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_78], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_241, reinterpret_tensor(buf435, (784, 384), (384, 1), 0), reinterpret_tensor(primals_240, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf436)
        del primals_241
        # Topologically Sorted Source Nodes: [x_262], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf437 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf436, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf436, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf436, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), None, True)
        buf438 = buf437[0]
        buf439 = buf437[1]
        buf440 = buf437[2]
        buf441 = buf437[3]
        del buf437
        buf442 = empty_strided_cuda((784, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_265], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf438, (784, 384), (384, 1), 0), reinterpret_tensor(primals_242, (384, 384), (1, 384), 0), out=buf442)
        buf443 = buf422; del buf422  # reuse
        buf447 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf448 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf551 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_258, x_268, x_269, layer_norm_39], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_29.run(buf443, buf430, primals_237, buf442, primals_243, primals_244, primals_245, buf447, buf448, buf551, 64, 384, grid=grid(64), stream=stream0)
        del primals_237
        del primals_243
        del primals_245
        buf449 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_247, reinterpret_tensor(buf448, (64, 384), (384, 1), 0), reinterpret_tensor(primals_246, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf449)
        del primals_247
        buf450 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_270], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf449, buf450, 98304, grid=grid(98304), stream=stream0)
        buf451 = buf430; del buf430  # reuse
        # Topologically Sorted Source Nodes: [x_271], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf450, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_248, (1536, 384), (1, 1536), 0), out=buf451)
        buf455 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf456 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf550 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_272, x_273], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_30.run(buf443, buf451, primals_249, primals_250, primals_251, buf455, buf456, buf550, 64, 384, grid=grid(64), stream=stream0)
        del primals_251
        buf457 = empty_strided_cuda((64, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_82], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_253, reinterpret_tensor(buf456, (64, 384), (384, 1), 0), reinterpret_tensor(primals_252, (384, 1152), (1, 384), 0), alpha=1, beta=1, out=buf457)
        del primals_253
        # Topologically Sorted Source Nodes: [x_274], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf458 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf457, (4, 4, 16, 96), (18432, 96, 1152, 1), 0), reinterpret_tensor(buf457, (4, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf457, (4, 4, 16, 96), (18432, 96, 1152, 1), 768), None, True)
        buf459 = buf458[0]
        buf460 = buf458[1]
        buf461 = buf458[2]
        buf462 = buf458[3]
        del buf458
        buf463 = empty_strided_cuda((64, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_277], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf459, (64, 384), (384, 1), 0), reinterpret_tensor(primals_254, (384, 384), (1, 384), 0), out=buf463)
        buf464 = buf443; del buf443  # reuse
        buf468 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf469 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf549 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_272, x_278, layer_norm_41], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_31.run(buf464, buf451, primals_249, buf463, primals_255, primals_256, primals_257, buf468, buf469, buf549, 64, 384, grid=grid(64), stream=stream0)
        del primals_249
        del primals_255
        del primals_257
        buf470 = empty_strided_cuda((64, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_84], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_259, reinterpret_tensor(buf469, (64, 384), (384, 1), 0), reinterpret_tensor(primals_258, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf470)
        del primals_259
        buf471 = empty_strided_cuda((4, 4, 4, 1536), (24576, 6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_279], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_26.run(buf470, buf471, 98304, grid=grid(98304), stream=stream0)
        buf472 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [x_280], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf471, (64, 1536), (1536, 1), 0), reinterpret_tensor(primals_260, (1536, 384), (1, 1536), 0), out=buf472)
        buf473 = buf464; del buf464  # reuse
        buf477 = reinterpret_tensor(buf451, (4, 4, 4, 384), (6144, 1536, 384, 1), 0); del buf451  # reuse
        buf478 = empty_strided_cuda((4, 4, 4, 384), (6144, 1536, 384, 1), torch.float32)
        buf548 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_281, x_282], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_32.run(buf473, buf472, primals_261, primals_262, primals_263, buf477, buf478, buf548, 64, 384, grid=grid(64), stream=stream0)
        del buf472
        del primals_261
        del primals_263
        buf479 = empty_strided_cuda((64, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_86], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_265, reinterpret_tensor(buf478, (64, 384), (384, 1), 0), reinterpret_tensor(primals_264, (384, 768), (1, 384), 0), alpha=1, beta=1, out=buf479)
        del primals_265
        buf481 = reinterpret_tensor(buf442, (4, 14, 14, 384), (75264, 5376, 384, 1), 0); del buf442  # reuse
        # Topologically Sorted Source Nodes: [x_286], Original ATen: [aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_33.run(buf478, buf481, 301056, grid=grid(301056), stream=stream0)
        buf482 = empty_strided_cuda((784, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_87], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_267, reinterpret_tensor(buf481, (784, 384), (384, 1), 0), reinterpret_tensor(primals_266, (384, 2304), (1, 384), 0), alpha=1, beta=1, out=buf482)
        del primals_267
        buf483 = empty_strided_cuda((4, 768, 7, 7), (37632, 1, 5376, 768), torch.int8)
        buf484 = empty_strided_cuda((4, 768, 7, 7), (37632, 1, 5376, 768), torch.float32)
        # Topologically Sorted Source Nodes: [x_289], Original ATen: [aten.max_pool2d_with_indices]
        stream0 = get_raw_stream(0)
        triton_poi_fused_max_pool2d_with_indices_34.run(buf482, buf483, buf484, 150528, grid=grid(150528), stream=stream0)
        # Topologically Sorted Source Nodes: [x_291], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf485 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf484, (4, 8, 49, 96), (37632, 96, 768, 1), 0), reinterpret_tensor(buf482, (4, 8, 196, 96), (451584, 96, 2304, 1), 768), reinterpret_tensor(buf482, (4, 8, 196, 96), (451584, 96, 2304, 1), 1536), None, True)
        buf486 = buf485[0]
        buf490 = empty_strided_cuda((196, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_294], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf486, (196, 768), (768, 1), 0), reinterpret_tensor(primals_268, (768, 768), (1, 768), 0), out=buf490)
        buf480 = empty_strided_cuda((4, 768, 2, 2), (3072, 1, 1536, 768), torch.int8)
        buf491 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf495 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf496 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf547 = empty_strided_cuda((4, 2, 2, 1), (4, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_284, x_297, x_298, layer_norm_43], Original ATen: [aten.max_pool2d_with_indices, aten.clone, aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_max_pool2d_with_indices_native_layer_norm_native_layer_norm_backward_35.run(buf479, buf490, primals_269, primals_270, primals_271, buf480, buf491, buf495, buf496, buf547, 16, 768, grid=grid(16), stream=stream0)
        del primals_269
        del primals_271
        buf487 = buf485[1]
        buf488 = buf485[2]
        buf489 = buf485[3]
        del buf485
        buf497 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_89], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_273, reinterpret_tensor(buf496, (16, 768), (768, 1), 0), reinterpret_tensor(primals_272, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf497)
        del primals_273
        buf498 = empty_strided_cuda((4, 2, 2, 3072), (12288, 6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_299], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_36.run(buf497, buf498, 49152, grid=grid(49152), stream=stream0)
        buf499 = empty_strided_cuda((16, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_300], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf498, (16, 3072), (3072, 1), 0), reinterpret_tensor(primals_274, (3072, 768), (1, 3072), 0), out=buf499)
        buf503 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf546 = empty_strided_cuda((4, 2, 2, 1), (4, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_301, x_302], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_37.run(buf491, buf499, primals_275, buf503, buf546, 16, 768, grid=grid(16), stream=stream0)
        buf504 = reinterpret_tensor(buf490, (4, 7, 7, 768), (37632, 5376, 768, 1), 0); del buf490  # reuse
        # Topologically Sorted Source Nodes: [x_302, x_303], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_38.run(buf503, primals_276, primals_277, buf504, 150528, grid=grid(150528), stream=stream0)
        del primals_277
        buf505 = empty_strided_cuda((196, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_91], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_279, reinterpret_tensor(buf504, (196, 768), (768, 1), 0), reinterpret_tensor(primals_278, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf505)
        del primals_279
        # Topologically Sorted Source Nodes: [x_305], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf506 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf505, (4, 8, 49, 96), (112896, 96, 2304, 1), 0), reinterpret_tensor(buf505, (4, 8, 49, 96), (112896, 96, 2304, 1), 768), reinterpret_tensor(buf505, (4, 8, 49, 96), (112896, 96, 2304, 1), 1536), None, True)
        buf507 = buf506[0]
        buf508 = buf506[1]
        buf509 = buf506[2]
        buf510 = buf506[3]
        del buf506
        buf511 = empty_strided_cuda((196, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_308], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf507, (196, 768), (768, 1), 0), reinterpret_tensor(primals_280, (768, 768), (1, 768), 0), out=buf511)
        buf512 = buf491; del buf491  # reuse
        buf516 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf517 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf545 = empty_strided_cuda((4, 2, 2, 1), (4, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_301, x_311, x_312, layer_norm_45], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_39.run(buf512, buf499, primals_275, buf511, primals_281, primals_282, primals_283, buf516, buf517, buf545, 16, 768, grid=grid(16), stream=stream0)
        del primals_275
        del primals_281
        del primals_283
        buf518 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_93], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_285, reinterpret_tensor(buf517, (16, 768), (768, 1), 0), reinterpret_tensor(primals_284, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf518)
        del primals_285
        buf519 = empty_strided_cuda((4, 2, 2, 3072), (12288, 6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_313], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_36.run(buf518, buf519, 49152, grid=grid(49152), stream=stream0)
        buf520 = buf499; del buf499  # reuse
        # Topologically Sorted Source Nodes: [x_314], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf519, (16, 3072), (3072, 1), 0), reinterpret_tensor(primals_286, (3072, 768), (1, 3072), 0), out=buf520)
        buf524 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf544 = empty_strided_cuda((4, 2, 2, 1), (4, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_315, x_316], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_37.run(buf512, buf520, primals_287, buf524, buf544, 16, 768, grid=grid(16), stream=stream0)
        buf525 = reinterpret_tensor(buf511, (4, 7, 7, 768), (37632, 5376, 768, 1), 0); del buf511  # reuse
        # Topologically Sorted Source Nodes: [x_316, x_317], Original ATen: [aten.native_layer_norm, aten.constant_pad_nd]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_native_layer_norm_38.run(buf524, primals_288, primals_289, buf525, 150528, grid=grid(150528), stream=stream0)
        del primals_289
        buf526 = empty_strided_cuda((196, 2304), (2304, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_95], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_291, reinterpret_tensor(buf525, (196, 768), (768, 1), 0), reinterpret_tensor(primals_290, (768, 2304), (1, 768), 0), alpha=1, beta=1, out=buf526)
        del primals_291
        # Topologically Sorted Source Nodes: [x_319], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf527 = torch.ops.aten._scaled_dot_product_efficient_attention.default(reinterpret_tensor(buf526, (4, 8, 49, 96), (112896, 96, 2304, 1), 0), reinterpret_tensor(buf526, (4, 8, 49, 96), (112896, 96, 2304, 1), 768), reinterpret_tensor(buf526, (4, 8, 49, 96), (112896, 96, 2304, 1), 1536), None, True)
        buf528 = buf527[0]
        buf529 = buf527[1]
        buf530 = buf527[2]
        buf531 = buf527[3]
        del buf527
        buf532 = empty_strided_cuda((196, 768), (768, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_322], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf528, (196, 768), (768, 1), 0), reinterpret_tensor(primals_292, (768, 768), (1, 768), 0), out=buf532)
        buf533 = buf512; del buf512  # reuse
        buf537 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf538 = empty_strided_cuda((4, 2, 2, 768), (3072, 1536, 768, 1), torch.float32)
        buf543 = empty_strided_cuda((4, 2, 2, 1), (4, 2, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_315, x_325, x_326, layer_norm_47], Original ATen: [aten.add, aten.clone, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_native_layer_norm_native_layer_norm_backward_39.run(buf533, buf520, primals_287, buf532, primals_293, primals_294, primals_295, buf537, buf538, buf543, 16, 768, grid=grid(16), stream=stream0)
        del buf532
        del primals_287
        del primals_293
        del primals_295
        buf539 = empty_strided_cuda((16, 3072), (3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_97], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_297, reinterpret_tensor(buf538, (16, 768), (768, 1), 0), reinterpret_tensor(primals_296, (768, 3072), (1, 768), 0), alpha=1, beta=1, out=buf539)
        del primals_297
        buf540 = empty_strided_cuda((4, 2, 2, 3072), (12288, 6144, 3072, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_327], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_36.run(buf539, buf540, 49152, grid=grid(49152), stream=stream0)
        buf541 = buf520; del buf520  # reuse
        # Topologically Sorted Source Nodes: [x_328], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf540, (16, 3072), (3072, 1), 0), reinterpret_tensor(primals_298, (3072, 768), (1, 3072), 0), out=buf541)
        buf542 = buf533; del buf533  # reuse
        # Topologically Sorted Source Nodes: [x_329], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_40.run(buf542, buf541, primals_299, 12288, grid=grid(12288), stream=stream0)
        del buf541
        del primals_299
    return (buf62, reinterpret_tensor(buf131, (4, 192, 8, 8), (12288, 1, 1536, 192), 0), reinterpret_tensor(buf473, (4, 384, 4, 4), (6144, 1, 1536, 384), 0), reinterpret_tensor(buf542, (4, 768, 2, 2), (3072, 1, 1536, 768), 0), primals_1, primals_3, primals_6, primals_12, primals_18, primals_24, primals_30, primals_38, primals_44, primals_50, primals_56, primals_62, primals_68, primals_76, primals_82, primals_88, primals_94, primals_100, primals_106, primals_112, primals_118, primals_124, primals_130, primals_136, primals_142, primals_148, primals_154, primals_160, primals_166, primals_172, primals_178, primals_184, primals_190, primals_196, primals_202, primals_208, primals_214, primals_220, primals_226, primals_232, primals_238, primals_244, primals_250, primals_256, primals_262, primals_270, primals_276, primals_282, primals_288, primals_294, buf1, buf2, buf3, buf4, buf5, buf8, buf11, buf14, buf22, reinterpret_tensor(buf23, (1024, 96), (96, 1), 0), reinterpret_tensor(buf24, (16, 1, 64, 96), (18432, 96, 288, 1), 0), reinterpret_tensor(buf24, (16, 1, 64, 96), (18432, 96, 288, 1), 96), reinterpret_tensor(buf24, (16, 1, 64, 96), (18432, 96, 288, 1), 192), buf26, buf27, buf28, buf29, buf35, reinterpret_tensor(buf36, (1024, 96), (96, 1), 0), buf37, reinterpret_tensor(buf38, (1024, 384), (384, 1), 0), buf43, reinterpret_tensor(buf44, (1024, 96), (96, 1), 0), reinterpret_tensor(buf45, (16, 1, 64, 96), (18432, 96, 288, 1), 0), reinterpret_tensor(buf45, (16, 1, 64, 96), (18432, 96, 288, 1), 96), reinterpret_tensor(buf45, (16, 1, 64, 96), (18432, 96, 288, 1), 192), buf47, buf48, buf49, buf50, buf56, reinterpret_tensor(buf57, (1024, 96), (96, 1), 0), buf58, reinterpret_tensor(buf59, (1024, 384), (384, 1), 0), buf66, reinterpret_tensor(buf67, (1024, 96), (96, 1), 0), reinterpret_tensor(buf68, (4, 192, 16, 16), (49152, 1, 3072, 192), 0), buf69, reinterpret_tensor(buf70, (1024, 96), (96, 1), 0), reinterpret_tensor(buf71, (16, 192, 8, 8), (36864, 1, 4608, 576), 0), buf72, reinterpret_tensor(buf73, (16, 2, 16, 96), (3072, 96, 192, 1), 0), reinterpret_tensor(buf71, (16, 2, 64, 96), (36864, 96, 576, 1), 192), reinterpret_tensor(buf71, (16, 2, 64, 96), (36864, 96, 576, 1), 384), buf75, buf76, buf77, buf78, buf84, reinterpret_tensor(buf85, (256, 192), (192, 1), 0), buf86, reinterpret_tensor(buf87, (256, 768), (768, 1), 0), buf92, reinterpret_tensor(buf93, (256, 192), (192, 1), 0), reinterpret_tensor(buf94, (16, 2, 16, 96), (9216, 96, 576, 1), 0), reinterpret_tensor(buf94, (16, 2, 16, 96), (9216, 96, 576, 1), 192), reinterpret_tensor(buf94, (16, 2, 16, 96), (9216, 96, 576, 1), 384), buf96, buf97, buf98, buf99, buf105, reinterpret_tensor(buf106, (256, 192), (192, 1), 0), buf107, reinterpret_tensor(buf108, (256, 768), (768, 1), 0), buf113, reinterpret_tensor(buf114, (256, 192), (192, 1), 0), reinterpret_tensor(buf115, (16, 2, 16, 96), (9216, 96, 576, 1), 0), reinterpret_tensor(buf115, (16, 2, 16, 96), (9216, 96, 576, 1), 192), reinterpret_tensor(buf115, (16, 2, 16, 96), (9216, 96, 576, 1), 384), buf117, buf118, buf119, buf120, buf126, reinterpret_tensor(buf127, (256, 192), (192, 1), 0), buf128, reinterpret_tensor(buf129, (256, 768), (768, 1), 0), buf135, reinterpret_tensor(buf136, (256, 192), (192, 1), 0), reinterpret_tensor(buf137, (4, 384, 8, 8), (24576, 1, 3072, 384), 0), buf138, reinterpret_tensor(buf139, (256, 192), (192, 1), 0), reinterpret_tensor(buf140, (16, 384, 4, 4), (18432, 1, 4608, 1152), 0), buf141, reinterpret_tensor(buf142, (16, 4, 4, 96), (1536, 96, 384, 1), 0), reinterpret_tensor(buf140, (16, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf140, (16, 4, 16, 96), (18432, 96, 1152, 1), 768), buf144, buf145, buf146, buf147, buf153, reinterpret_tensor(buf154, (64, 384), (384, 1), 0), buf155, reinterpret_tensor(buf156, (64, 1536), (1536, 1), 0), buf161, reinterpret_tensor(buf162, (784, 384), (384, 1), 0), reinterpret_tensor(buf163, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf163, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf163, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf165, buf166, buf167, buf168, buf174, reinterpret_tensor(buf175, (64, 384), (384, 1), 0), buf176, reinterpret_tensor(buf177, (64, 1536), (1536, 1), 0), buf182, reinterpret_tensor(buf183, (784, 384), (384, 1), 0), reinterpret_tensor(buf184, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf184, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf184, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf186, buf187, buf188, buf189, buf195, reinterpret_tensor(buf196, (64, 384), (384, 1), 0), buf197, reinterpret_tensor(buf198, (64, 1536), (1536, 1), 0), buf203, reinterpret_tensor(buf204, (784, 384), (384, 1), 0), reinterpret_tensor(buf205, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf205, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf205, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf207, buf208, buf209, buf210, buf216, reinterpret_tensor(buf217, (64, 384), (384, 1), 0), buf218, reinterpret_tensor(buf219, (64, 1536), (1536, 1), 0), buf224, reinterpret_tensor(buf225, (784, 384), (384, 1), 0), reinterpret_tensor(buf226, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf226, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf226, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf228, buf229, buf230, buf231, buf237, reinterpret_tensor(buf238, (64, 384), (384, 1), 0), buf239, reinterpret_tensor(buf240, (64, 1536), (1536, 1), 0), buf245, reinterpret_tensor(buf246, (784, 384), (384, 1), 0), reinterpret_tensor(buf247, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf247, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf247, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf249, buf250, buf251, buf252, buf258, reinterpret_tensor(buf259, (64, 384), (384, 1), 0), buf260, reinterpret_tensor(buf261, (64, 1536), (1536, 1), 0), buf266, reinterpret_tensor(buf267, (784, 384), (384, 1), 0), reinterpret_tensor(buf268, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf268, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf268, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf270, buf271, buf272, buf273, buf279, reinterpret_tensor(buf280, (64, 384), (384, 1), 0), buf281, reinterpret_tensor(buf282, (64, 1536), (1536, 1), 0), buf287, reinterpret_tensor(buf288, (64, 384), (384, 1), 0), reinterpret_tensor(buf289, (4, 4, 16, 96), (18432, 96, 1152, 1), 0), reinterpret_tensor(buf289, (4, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf289, (4, 4, 16, 96), (18432, 96, 1152, 1), 768), buf291, buf292, buf293, buf294, buf300, reinterpret_tensor(buf301, (64, 384), (384, 1), 0), buf302, reinterpret_tensor(buf303, (64, 1536), (1536, 1), 0), buf308, reinterpret_tensor(buf309, (784, 384), (384, 1), 0), reinterpret_tensor(buf310, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf310, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf310, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf312, buf313, buf314, buf315, buf321, reinterpret_tensor(buf322, (64, 384), (384, 1), 0), buf323, reinterpret_tensor(buf324, (64, 1536), (1536, 1), 0), buf329, reinterpret_tensor(buf330, (784, 384), (384, 1), 0), reinterpret_tensor(buf331, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf331, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf331, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf333, buf334, buf335, buf336, buf342, reinterpret_tensor(buf343, (64, 384), (384, 1), 0), buf344, reinterpret_tensor(buf345, (64, 1536), (1536, 1), 0), buf350, reinterpret_tensor(buf351, (784, 384), (384, 1), 0), reinterpret_tensor(buf352, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf352, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf352, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf354, buf355, buf356, buf357, buf363, reinterpret_tensor(buf364, (64, 384), (384, 1), 0), buf365, reinterpret_tensor(buf366, (64, 1536), (1536, 1), 0), buf371, reinterpret_tensor(buf372, (64, 384), (384, 1), 0), reinterpret_tensor(buf373, (4, 4, 16, 96), (18432, 96, 1152, 1), 0), reinterpret_tensor(buf373, (4, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf373, (4, 4, 16, 96), (18432, 96, 1152, 1), 768), buf375, buf376, buf377, buf378, buf384, reinterpret_tensor(buf385, (64, 384), (384, 1), 0), buf386, reinterpret_tensor(buf387, (64, 1536), (1536, 1), 0), buf392, reinterpret_tensor(buf393, (784, 384), (384, 1), 0), reinterpret_tensor(buf394, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf394, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf394, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf396, buf397, buf398, buf399, buf405, reinterpret_tensor(buf406, (64, 384), (384, 1), 0), buf407, reinterpret_tensor(buf408, (64, 1536), (1536, 1), 0), buf413, reinterpret_tensor(buf414, (784, 384), (384, 1), 0), reinterpret_tensor(buf415, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf415, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf415, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf417, buf418, buf419, buf420, buf426, reinterpret_tensor(buf427, (64, 384), (384, 1), 0), buf428, reinterpret_tensor(buf429, (64, 1536), (1536, 1), 0), buf434, reinterpret_tensor(buf435, (784, 384), (384, 1), 0), reinterpret_tensor(buf436, (4, 4, 196, 96), (225792, 96, 1152, 1), 0), reinterpret_tensor(buf436, (4, 4, 196, 96), (225792, 96, 1152, 1), 384), reinterpret_tensor(buf436, (4, 4, 196, 96), (225792, 96, 1152, 1), 768), buf438, buf439, buf440, buf441, buf447, reinterpret_tensor(buf448, (64, 384), (384, 1), 0), buf449, reinterpret_tensor(buf450, (64, 1536), (1536, 1), 0), buf455, reinterpret_tensor(buf456, (64, 384), (384, 1), 0), reinterpret_tensor(buf457, (4, 4, 16, 96), (18432, 96, 1152, 1), 0), reinterpret_tensor(buf457, (4, 4, 16, 96), (18432, 96, 1152, 1), 384), reinterpret_tensor(buf457, (4, 4, 16, 96), (18432, 96, 1152, 1), 768), buf459, buf460, buf461, buf462, buf468, reinterpret_tensor(buf469, (64, 384), (384, 1), 0), buf470, reinterpret_tensor(buf471, (64, 1536), (1536, 1), 0), buf477, reinterpret_tensor(buf478, (64, 384), (384, 1), 0), reinterpret_tensor(buf479, (4, 768, 4, 4), (12288, 1, 3072, 768), 0), buf480, reinterpret_tensor(buf481, (784, 384), (384, 1), 0), reinterpret_tensor(buf482, (4, 768, 14, 14), (451584, 1, 32256, 2304), 0), buf483, reinterpret_tensor(buf484, (4, 8, 49, 96), (37632, 96, 768, 1), 0), reinterpret_tensor(buf482, (4, 8, 196, 96), (451584, 96, 2304, 1), 768), reinterpret_tensor(buf482, (4, 8, 196, 96), (451584, 96, 2304, 1), 1536), buf486, buf487, buf488, buf489, buf495, reinterpret_tensor(buf496, (16, 768), (768, 1), 0), buf497, reinterpret_tensor(buf498, (16, 3072), (3072, 1), 0), buf503, reinterpret_tensor(buf504, (196, 768), (768, 1), 0), reinterpret_tensor(buf505, (4, 8, 49, 96), (112896, 96, 2304, 1), 0), reinterpret_tensor(buf505, (4, 8, 49, 96), (112896, 96, 2304, 1), 768), reinterpret_tensor(buf505, (4, 8, 49, 96), (112896, 96, 2304, 1), 1536), buf507, buf508, buf509, buf510, buf516, reinterpret_tensor(buf517, (16, 768), (768, 1), 0), buf518, reinterpret_tensor(buf519, (16, 3072), (3072, 1), 0), buf524, reinterpret_tensor(buf525, (196, 768), (768, 1), 0), reinterpret_tensor(buf526, (4, 8, 49, 96), (112896, 96, 2304, 1), 0), reinterpret_tensor(buf526, (4, 8, 49, 96), (112896, 96, 2304, 1), 768), reinterpret_tensor(buf526, (4, 8, 49, 96), (112896, 96, 2304, 1), 1536), buf528, buf529, buf530, buf531, buf537, reinterpret_tensor(buf538, (16, 768), (768, 1), 0), buf539, reinterpret_tensor(buf540, (16, 3072), (3072, 1), 0), primals_298, primals_296, buf543, primals_292, primals_290, buf544, primals_286, primals_284, buf545, primals_280, primals_278, buf546, primals_274, primals_272, buf547, primals_268, primals_266, primals_264, buf548, primals_260, primals_258, buf549, primals_254, primals_252, buf550, primals_248, primals_246, buf551, primals_242, primals_240, buf552, primals_236, primals_234, buf553, primals_230, primals_228, buf554, primals_224, primals_222, buf555, primals_218, primals_216, buf556, primals_212, primals_210, buf557, primals_206, primals_204, buf558, primals_200, primals_198, buf559, primals_194, primals_192, buf560, primals_188, primals_186, buf561, primals_182, primals_180, buf562, primals_176, primals_174, buf563, primals_170, primals_168, buf564, primals_164, primals_162, buf565, primals_158, primals_156, buf566, primals_152, primals_150, buf567, primals_146, primals_144, buf568, primals_140, primals_138, buf569, primals_134, primals_132, buf570, primals_128, primals_126, buf571, primals_122, primals_120, buf572, primals_116, primals_114, buf573, primals_110, primals_108, buf574, primals_104, primals_102, buf575, primals_98, primals_96, buf576, primals_92, primals_90, buf577, primals_86, primals_84, buf578, primals_80, primals_78, buf579, primals_74, primals_72, primals_70, buf580, primals_66, primals_64, buf581, primals_60, primals_58, buf582, primals_54, primals_52, buf583, primals_48, primals_46, buf584, primals_42, primals_40, buf585, primals_36, primals_34, primals_32, buf586, primals_28, primals_26, buf587, primals_22, primals_20, buf588, primals_16, primals_14, buf589, primals_10, primals_8, buf590, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((96, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 96, 8, 8), (6144, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 96, 14, 14), (18816, 196, 14, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((288, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((288, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((96, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((384, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((96, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((576, 96), (96, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((192, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1152, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((2304, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((2304, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((768, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((768, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
