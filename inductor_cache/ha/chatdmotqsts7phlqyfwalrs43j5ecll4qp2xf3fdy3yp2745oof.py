# AOT ID: ['201_forward']
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


# kernel path: inductor_cache/oo/cooqdjg57nm7ab2dp6u7gyehmqq43hwjzwo54iqde5mfc5vrapdw.py
# Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   input_287 => clamp_max_69, clamp_min_67, clamp_min_69, convert_element_type_214, iota, mul_322, sub_107
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (4,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_214 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota, torch.float32), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_214, 0.3333333333333333), kwargs = {})
#   %clamp_min_67 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_322, 0.0), kwargs = {})
#   %sub_107 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_67, %convert_element_type_217), kwargs = {})
#   %clamp_min_69 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_107, 0.0), kwargs = {})
#   %clamp_max_69 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_69, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_0 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_0', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hg/chgipuneekvmk3vfughnn2fgcych5blsttbpy6utjo7wpgxm7wik.py
# Topologically Sorted Source Nodes: [input_287], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_287 => convert_element_type_215
# Graph fragment:
#   %convert_element_type_215 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.int64), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ls/clskupd3nfqseasdggk5heuretgeuvvly25hnqcp3nhosltyhjg5.py
# Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   input_287 => add_232, clamp_max_67
# Graph fragment:
#   %add_232 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_215, 1), kwargs = {})
#   %clamp_max_67 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_232, 1), kwargs = {})
triton_poi_fused_add_clamp_2 = async_compile.triton('triton_poi_fused_add_clamp_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.3333333333333333
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.minimum(tmp8, tmp7)
    tl.store(out_ptr0 + (x0), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/33/c33wrqck5gzjdkrijnx3o6nkeii72zizxdlngh3p2be2ic2ilxye.py
# Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
# Source node to ATen node mapping:
#   input_320 => clamp_max_81, clamp_min_79, clamp_min_81, convert_element_type_242, iota_2, mul_363, sub_124
# Graph fragment:
#   %iota_2 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %convert_element_type_242 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%iota_2, torch.float32), kwargs = {})
#   %mul_363 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_242, 0.42857142857142855), kwargs = {})
#   %clamp_min_79 : [num_users=3] = call_function[target=torch.ops.aten.clamp_min.default](args = (%mul_363, 0.0), kwargs = {})
#   %sub_124 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_min_79, %convert_element_type_245), kwargs = {})
#   %clamp_min_81 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%sub_124, 0.0), kwargs = {})
#   %clamp_max_81 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_81, 1.0), kwargs = {})
triton_poi_fused__to_copy_arange_clamp_mul_sub_3 = async_compile.triton('triton_poi_fused__to_copy_arange_clamp_mul_sub_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_arange_clamp_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_arange_clamp_mul_sub_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 - tmp7
    tmp9 = triton_helpers.maximum(tmp8, tmp4)
    tmp10 = 1.0
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6v/c6v5exesdkarh7s6d423lbooj4ha5kljdlsy7krelbsu7kmpc4md.py
# Topologically Sorted Source Nodes: [input_320], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_320 => convert_element_type_243
# Graph fragment:
#   %convert_element_type_243 : [num_users=5] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.int64), kwargs = {})
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ka/ckavcf7daqxap55wwt4uiqukbllubusogffb26xpy46vu3yhjq2x.py
# Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.add, aten.clamp]
# Source node to ATen node mapping:
#   input_320 => add_263, clamp_max_79
# Graph fragment:
#   %add_263 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_243, 1), kwargs = {})
#   %clamp_max_79 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%add_263, 3), kwargs = {})
triton_poi_fused_add_clamp_5 = async_compile.triton('triton_poi_fused_add_clamp_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.42857142857142855
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.int32)
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp6 + tmp7
    tmp9 = tl.full([1], 3, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/53/c53xc5a6obyvyqxih4ioqwihtvlyusb3vhgfr2jc44gslpeq6dcs.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/a7/ca7xguimf6gszcga5mxtn6hr4yxobglrdev4shb2jfvykvberdmi.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_7 = async_compile.triton('triton_poi_fused_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 12)
    y1 = yindex // 12
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 12*x2 + 108*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/to/ctoh7nkyfjxi62sctjwu2wihjhxmtwsa2ye7zs22mz4rwvdrfm4r.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_8 = async_compile.triton('triton_poi_fused_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/r3/cr3mctbvtjemxi6gg3s6na45uslblz7vvltscbatoxkidywlimm6.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => clamp_max, clamp_min
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 12)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/uz/cuzgq6nkton4oznrywpfw4qchvg4tpn37nksn6hjyioiyndhpuqt.py
# Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_4 => convolution_1
#   input_5 => add_3, mul_4, mul_5, sub_1
#   input_6 => clamp_max_1, clamp_min_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max, %primals_8, %primals_9, [2, 2], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_3, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ro/crowahkzxg3rnx5znkkfxoxjlfyuyozlxdl5srdaaarypvoxp7iv.py
# Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_7 => convolution_2
#   input_8 => add_5, mul_7, mul_8, sub_2
#   input_9 => clamp_max_2, clamp_min_2
# Graph fragment:
#   %convolution_2 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_1, %primals_14, %primals_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_5, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 7)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/ro/crot4jx26j5h7slziyrucbifv2lj56i7mavlzbb47g4zefsy2voe.py
# Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_10 => convolution_3
#   input_11 => add_7, mul_10, mul_11, sub_3
#   input_12 => clamp_max_3, clamp_min_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_2, %primals_20, %primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 172032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 42)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxv27cxakc7r3u3l74murtujnhm3452ovq3i4ibe3tl7xdn3hvty.py
# Topologically Sorted Source Nodes: [input_16, input_17, input_18, input_19, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_16 => convolution_5
#   input_17 => add_11, mul_16, mul_17, sub_5
#   input_18 => convolution_6
#   input_19 => add_13, mul_19, mul_20, sub_6
#   out => add_14
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_4, %primals_32, %primals_33, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %convolution_6 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_1, %primals_38, %primals_39, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %add_13), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_13(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), None)
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(in_out_ptr1 + (x2), tmp5, None)
    tl.store(out_ptr0 + (x2), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/qm/cqmkawlsyizf6phwyeydl7j42mmgc7b32dbsiysvuct752phla63.py
# Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_20 => convolution_7
#   input_21 => add_16, mul_22, mul_23, sub_7
#   input_22 => clamp_max_5, clamp_min_5
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_14, %primals_44, %primals_45, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %clamp_min_5 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_16, 0.0), kwargs = {})
#   %clamp_max_5 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_5, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/mj/cmj3vxtypawq73336xn24jfayj4yy6rk2z7syg2lah7ak2uakhsv.py
# Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_23 => convolution_8
#   input_24 => add_18, mul_25, mul_26, sub_8
#   input_25 => clamp_max_6, clamp_min_6
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_5, %primals_50, %primals_51, [2, 2], [1, 1], [1, 1], False, [0, 0], 144), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %clamp_min_6 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_18, 0.0), kwargs = {})
#   %clamp_max_6 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_6, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 144)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/v5/cv5iwno2t6mumrn75mlaoq5gliscp4kvoqbuf2ygitej66xv7skq.py
# Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_26 => convolution_9
#   input_27 => add_20, mul_28, mul_29, sub_9
# Graph fragment:
#   %convolution_9 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_6, %primals_56, %primals_57, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_9, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_20 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_16', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_16', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 71680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 70)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pu/cpuphq3g4knt654k7ivchhqenmpe4cjstr2ec2mg6qtjginafwyj.py
# Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_28 => convolution_10
#   input_29 => add_22, mul_31, mul_32, sub_10
#   input_30 => clamp_max_7, clamp_min_7
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_20, %primals_62, %primals_63, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %clamp_min_7 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_22, 0.0), kwargs = {})
#   %clamp_max_7 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_7, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 25)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y2/cy23ot4xglb45ekprodvecpalbdq66iciml7q7tffvphew3wv526.py
# Topologically Sorted Source Nodes: [input_31, input_32, input_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_31 => convolution_11
#   input_32 => add_24, mul_34, mul_35, sub_11
#   input_33 => clamp_max_8, clamp_min_8
# Graph fragment:
#   %convolution_11 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_7, %primals_68, %primals_69, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %clamp_min_8 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_24, 0.0), kwargs = {})
#   %clamp_max_8 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_8, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 153600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 150)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2i/c2iul5mraaaxa5r3ocpjbym4oltt6vlwsz3cj4vidorr3ockc6up.py
# Topologically Sorted Source Nodes: [input_37, input_38, input_39, input_40, out_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_37 => convolution_13
#   input_38 => add_28, mul_40, mul_41, sub_13
#   input_39 => convolution_14
#   input_40 => add_30, mul_43, mul_44, sub_14
#   out_1 => add_31
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_9, %primals_80, %primals_81, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %convolution_14 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_20, %primals_86, %primals_87, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %add_31 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_28, %add_30), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 71680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 70)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lx/clxfvjkqnmkto4bpqvvwakx3vclfjp76vmkcrg5nbmnwchut7nn7.py
# Topologically Sorted Source Nodes: [input_41, input_42, input_43], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_41 => convolution_15
#   input_42 => add_33, mul_46, mul_47, sub_15
#   input_43 => clamp_max_10, clamp_min_10
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_31, %primals_92, %primals_93, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %clamp_min_10 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_33, 0.0), kwargs = {})
#   %clamp_max_10 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_10, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 24)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/mn/cmnadgdg75rhd4t6qaajqi6qbj4pougy4mta3kwbixxyajmm4hne.py
# Topologically Sorted Source Nodes: [input_54, input_55, input_56], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_54 => convolution_20
#   input_55 => add_44, mul_61, mul_62, sub_20
#   input_56 => clamp_max_13, clamp_min_13
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_42, %primals_122, %primals_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_44 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
#   %clamp_min_13 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_44, 0.0), kwargs = {})
#   %clamp_max_13 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_13, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 430080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 420)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/di/cdidzyym3hvyin4qsbngly5ejiveqnnq773rcvf5s6jlj6u4d4ll.py
# Topologically Sorted Source Nodes: [input_57, input_58, input_59], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_57 => convolution_21
#   input_58 => add_46, mul_64, mul_65, sub_21
#   input_59 => clamp_max_14, clamp_min_14
# Graph fragment:
#   %convolution_21 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_13, %primals_128, %primals_129, [2, 2], [1, 1], [1, 1], False, [0, 0], 420), kwargs = {})
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_46, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 107520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 420)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cobq4w4ymxwfm2r6ubqtxlyvx6uvrm2gvgy3w3b4qd5tncu2uk6z.py
# Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_60 => convolution_22
#   input_61 => add_48, mul_67, mul_68, sub_22
# Graph fragment:
#   %convolution_22 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_14, %primals_134, %primals_135, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_23', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 150)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gh/cghlayodqcuitmxzc5sbwiwlgkh6g2tjlahsk6n6sscdegcwg7uy.py
# Topologically Sorted Source Nodes: [input_62, input_63, input_64], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_62 => convolution_23
#   input_63 => add_50, mul_70, mul_71, sub_23
#   input_64 => clamp_max_15, clamp_min_15
# Graph fragment:
#   %convolution_23 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_48, %primals_140, %primals_141, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_189), kwargs = {})
#   %add_50 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_191), kwargs = {})
#   %clamp_min_15 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_50, 0.0), kwargs = {})
#   %clamp_max_15 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_15, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 56)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/v2/cv2jraozu4oicybyazgf5xjtsy5nvn6wktbv4swvrgtwkjs65e7g.py
# Topologically Sorted Source Nodes: [input_65, input_66, input_67], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_65 => convolution_24
#   input_66 => add_52, mul_73, mul_74, sub_24
#   input_67 => clamp_max_16, clamp_min_16
# Graph fragment:
#   %convolution_24 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_15, %primals_146, %primals_147, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %clamp_min_16 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_52, 0.0), kwargs = {})
#   %clamp_max_16 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_16, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 86016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 336)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, None)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/57/c577zqkkl3qxfjji7yt2hq2rhxjbnj6qrn3knabjqeejj7digdyx.py
# Topologically Sorted Source Nodes: [input_71, input_72, input_73, input_74, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_71 => convolution_26
#   input_72 => add_56, mul_79, mul_80, sub_26
#   input_73 => convolution_27
#   input_74 => add_58, mul_82, mul_83, sub_27
#   out_3 => add_59
# Graph fragment:
#   %convolution_26 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_17, %primals_158, %primals_159, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_26 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_26, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_26, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %convolution_27 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_48, %primals_164, %primals_165, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_27, %unsqueeze_217), kwargs = {})
#   %mul_82 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_219), kwargs = {})
#   %mul_83 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_82, %unsqueeze_221), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_83, %unsqueeze_223), kwargs = {})
#   %add_59 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_56, %add_58), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 150)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vj/cvjl43j2xwpyspxxz6omfrwgzl6qhtp3ue4cffqiwk64vkppuexm.py
# Topologically Sorted Source Nodes: [input_75, input_76, input_77], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_75 => convolution_28
#   input_76 => add_61, mul_85, mul_86, sub_28
#   input_77 => clamp_max_18, clamp_min_18
# Graph fragment:
#   %convolution_28 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_59, %primals_170, %primals_171, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %clamp_min_18 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_61, 0.0), kwargs = {})
#   %clamp_max_18 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_18, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 150)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bt/cbtl5kzs67t42v224nkhwa5m3nlscoe3maekwcrav6tc6pp4urgi.py
# Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   adaptive_avg_pool2d => mean
# Graph fragment:
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%clamp_max_18, [-1, -2], True), kwargs = {})
triton_per_fused_mean_28 = async_compile.triton('triton_per_fused_mean_28', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_mean_28(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 600
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 150)
    x1 = xindex // 150
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 150*r2 + 9600*x1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ps/cpsyz46qhtjsg4qbinzs2pi4fn3prez7insshs7fqbvpgnnfcacj.py
# Topologically Sorted Source Nodes: [input_78, input_79], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   input_78 => add_tensor
#   input_79 => relu
# Graph fragment:
#   %add_tensor : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default, %primals_177), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor,), kwargs = {})
triton_poi_fused_addmm_relu_29 = async_compile.triton('triton_poi_fused_addmm_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_29(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 18)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7f/c7fw277z7nhsgksqofrxo53knbfl5aoed7lmsvpltbstkigucdrd.py
# Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   input_82 => mul_87
# Graph fragment:
#   %mul_87 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max_18, %expand), kwargs = {})
triton_poi_fused_mul_30 = async_compile.triton('triton_poi_fused_mul_30', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_30', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_30(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 150)
    x2 = xindex // 9600
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 150*x2), xmask, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tl.store(in_out_ptr0 + (x3), tmp3, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wc/cwc6iusbxqncvp6vv3jdyc3ikv6tvtjj76z54klfpuokrolzkhma.py
# Topologically Sorted Source Nodes: [input_83, input_84, input_85], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_83 => convolution_29
#   input_84 => add_63, mul_89, mul_90, sub_29
#   input_85 => clamp_max_19, clamp_min_19
# Graph fragment:
#   %convolution_29 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%mul_87, %primals_180, %primals_181, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_29, %unsqueeze_233), kwargs = {})
#   %mul_89 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_235), kwargs = {})
#   %mul_90 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_89, %unsqueeze_237), kwargs = {})
#   %add_63 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_90, %unsqueeze_239), kwargs = {})
#   %clamp_min_19 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_63, 0.0), kwargs = {})
#   %clamp_max_19 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_19, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 73)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jb/cjb2exemahjuizl4s6lhiufgv56nb45hwuxe4ngckzc5foefb4tn.py
# Topologically Sorted Source Nodes: [input_86, input_87, input_88], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_86 => convolution_30
#   input_87 => add_65, mul_92, mul_93, sub_30
#   input_88 => clamp_max_20, clamp_min_20
# Graph fragment:
#   %convolution_30 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_19, %primals_186, %primals_187, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_30 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_30, %unsqueeze_241), kwargs = {})
#   %mul_92 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_30, %unsqueeze_243), kwargs = {})
#   %mul_93 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_92, %unsqueeze_245), kwargs = {})
#   %add_65 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_93, %unsqueeze_247), kwargs = {})
#   %clamp_min_20 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_65, 0.0), kwargs = {})
#   %clamp_max_20 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_20, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 438)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vo/cvo4m4iwb7jrfcfxgr3ciz2h7knx2bw25q3tvoqf6h4n3tqlwqlm.py
# Topologically Sorted Source Nodes: [input_96, input_97, input_98], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_96 => convolution_34
#   input_97 => add_74, mul_104, mul_105, sub_34
#   input_98 => clamp_max_22, clamp_min_22
# Graph fragment:
#   %convolution_34 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_72, %primals_210, %primals_211, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_34, %unsqueeze_273), kwargs = {})
#   %mul_104 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_34, %unsqueeze_275), kwargs = {})
#   %mul_105 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_104, %unsqueeze_277), kwargs = {})
#   %add_74 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_105, %unsqueeze_279), kwargs = {})
#   %clamp_min_22 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_74, 0.0), kwargs = {})
#   %clamp_max_22 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_22, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 71)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dv/cdvlmgkguj27cnrandluk3d2llg7vtioxx62w5xbflfqxmbmykix.py
# Topologically Sorted Source Nodes: [input_99, input_100, input_101], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_100 => add_76, mul_107, mul_108, sub_35
#   input_101 => clamp_max_23, clamp_min_23
#   input_99 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_22, %primals_216, %primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_35 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_35, %unsqueeze_281), kwargs = {})
#   %mul_107 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_35, %unsqueeze_283), kwargs = {})
#   %mul_108 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_107, %unsqueeze_285), kwargs = {})
#   %add_76 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_108, %unsqueeze_287), kwargs = {})
#   %clamp_min_23 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_76, 0.0), kwargs = {})
#   %clamp_max_23 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_23, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 109056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 426)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/o5/co5pgijzdexw262nfmwlngethdknpzhkpnmc4k6fxvqazvvadjmq.py
# Topologically Sorted Source Nodes: [input_109, input_110, input_111], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_109 => convolution_39
#   input_110 => add_85, mul_119, mul_120, sub_39
#   input_111 => clamp_max_25, clamp_min_25
# Graph fragment:
#   %convolution_39 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_83, %primals_240, %primals_241, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_120 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_119, %unsqueeze_317), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_120, %unsqueeze_319), kwargs = {})
#   %clamp_min_25 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_85, 0.0), kwargs = {})
#   %clamp_max_25 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_25, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 75)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wr/cwrxldqpykchmhzy3zjwiyy6ql2pjszmluon6gxvqhh6tigm7m6a.py
# Topologically Sorted Source Nodes: [input_112, input_113, input_114], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_112 => convolution_40
#   input_113 => add_87, mul_122, mul_123, sub_40
#   input_114 => clamp_max_26, clamp_min_26
# Graph fragment:
#   %convolution_40 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_25, %primals_246, %primals_247, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_40, %unsqueeze_321), kwargs = {})
#   %mul_122 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_40, %unsqueeze_323), kwargs = {})
#   %mul_123 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_122, %unsqueeze_325), kwargs = {})
#   %add_87 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_123, %unsqueeze_327), kwargs = {})
#   %clamp_min_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_87, 0.0), kwargs = {})
#   %clamp_max_26 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_26, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 450)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zg/czgjh25e4pf334q6inypz6b33uqhv76gh3n4gd7u4u5qujchyicl.py
# Topologically Sorted Source Nodes: [input_122, input_123, input_124], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_122 => convolution_44
#   input_123 => add_96, mul_134, mul_135, sub_44
#   input_124 => clamp_max_28, clamp_min_28
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_94, %primals_270, %primals_271, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_44, %unsqueeze_353), kwargs = {})
#   %mul_134 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_44, %unsqueeze_355), kwargs = {})
#   %mul_135 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_134, %unsqueeze_357), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_135, %unsqueeze_359), kwargs = {})
#   %clamp_min_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_96, 0.0), kwargs = {})
#   %clamp_max_28 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_28, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 900)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dh/cdhqw5lqqlp2ptam5xxml5hhh2q6bbfga3drxkysle4675nwpf2g.py
# Topologically Sorted Source Nodes: [input_125, input_126, input_127], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_125 => convolution_45
#   input_126 => add_98, mul_137, mul_138, sub_45
#   input_127 => clamp_max_29, clamp_min_29
# Graph fragment:
#   %convolution_45 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_28, %primals_276, %primals_277, [2, 2], [1, 1], [1, 1], False, [0, 0], 900), kwargs = {})
#   %sub_45 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_45, %unsqueeze_361), kwargs = {})
#   %mul_137 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_45, %unsqueeze_363), kwargs = {})
#   %mul_138 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_137, %unsqueeze_365), kwargs = {})
#   %add_98 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_138, %unsqueeze_367), kwargs = {})
#   %clamp_min_29 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_98, 0.0), kwargs = {})
#   %clamp_max_29 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_29, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_38 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_38', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_38', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 57600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 900)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xv/cxv7o5vu5ctzapf7whs24ndnmqsn4b2qpxihvtd7dzfengdlxbr6.py
# Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_128 => convolution_46
#   input_129 => add_100, mul_140, mul_141, sub_46
# Graph fragment:
#   %convolution_46 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_29, %primals_282, %primals_283, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_46, %unsqueeze_369), kwargs = {})
#   %mul_140 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_371), kwargs = {})
#   %mul_141 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_140, %unsqueeze_373), kwargs = {})
#   %add_100 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_141, %unsqueeze_375), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_39 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_39', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_39', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 325)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/34/c34kjexwprpluyz3si2kpwp5sp6vv7wyseuxzufbf6jkwumjvz4k.py
# Topologically Sorted Source Nodes: [input_130, input_131, input_132], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_130 => convolution_47
#   input_131 => add_102, mul_143, mul_144, sub_47
#   input_132 => clamp_max_30, clamp_min_30
# Graph fragment:
#   %convolution_47 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_100, %primals_288, %primals_289, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_47 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_47, %unsqueeze_377), kwargs = {})
#   %mul_143 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_47, %unsqueeze_379), kwargs = {})
#   %mul_144 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_143, %unsqueeze_381), kwargs = {})
#   %add_102 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_144, %unsqueeze_383), kwargs = {})
#   %clamp_min_30 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_102, 0.0), kwargs = {})
#   %clamp_max_30 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_30, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 132)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yz/cyzgzm5ueppwzmq5byajq2wjonrf7gn4ifdylfs5lykervep3m5n.py
# Topologically Sorted Source Nodes: [input_133, input_134, input_135], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_133 => convolution_48
#   input_134 => add_104, mul_146, mul_147, sub_48
#   input_135 => clamp_max_31, clamp_min_31
# Graph fragment:
#   %convolution_48 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_30, %primals_294, %primals_295, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_48, %unsqueeze_385), kwargs = {})
#   %mul_146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_48, %unsqueeze_387), kwargs = {})
#   %mul_147 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_146, %unsqueeze_389), kwargs = {})
#   %add_104 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_147, %unsqueeze_391), kwargs = {})
#   %clamp_min_31 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_104, 0.0), kwargs = {})
#   %clamp_max_31 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_31, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_41 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_41', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_41', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_41(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 792)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ml/cml3gry2pmwa5sowlfdzszz2m7awrkd7i5aud7xv44hrqcy77wea.py
# Topologically Sorted Source Nodes: [input_139, input_140, input_141, input_142, out_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_139 => convolution_50
#   input_140 => add_108, mul_152, mul_153, sub_50
#   input_141 => convolution_51
#   input_142 => add_110, mul_155, mul_156, sub_51
#   out_7 => add_111
# Graph fragment:
#   %convolution_50 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_32, %primals_306, %primals_307, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_50, %unsqueeze_401), kwargs = {})
#   %mul_152 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_50, %unsqueeze_403), kwargs = {})
#   %mul_153 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_152, %unsqueeze_405), kwargs = {})
#   %add_108 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_153, %unsqueeze_407), kwargs = {})
#   %convolution_51 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_100, %primals_312, %primals_313, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_51 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_51, %unsqueeze_409), kwargs = {})
#   %mul_155 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_51, %unsqueeze_411), kwargs = {})
#   %mul_156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_155, %unsqueeze_413), kwargs = {})
#   %add_110 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_156, %unsqueeze_415), kwargs = {})
#   %add_111 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_108, %add_110), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 325)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zc/czcgvvpu7ld3btraynv75b4asoyev5kuioq7d2ve64ncdnp2bjmk.py
# Topologically Sorted Source Nodes: [input_143, input_144, input_145], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_143 => convolution_52
#   input_144 => add_113, mul_158, mul_159, sub_52
#   input_145 => clamp_max_33, clamp_min_33
# Graph fragment:
#   %convolution_52 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_111, %primals_318, %primals_319, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_52, %unsqueeze_417), kwargs = {})
#   %mul_158 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_52, %unsqueeze_419), kwargs = {})
#   %mul_159 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_158, %unsqueeze_421), kwargs = {})
#   %add_113 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_159, %unsqueeze_423), kwargs = {})
#   %clamp_min_33 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_113, 0.0), kwargs = {})
#   %clamp_max_33 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_33, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 124)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nw/cnwfyvgp2zh7hdaev5c7pue46njy6dhp4pfg76a4x5bg5tgmifqu.py
# Topologically Sorted Source Nodes: [input_146, input_147, input_148], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_146 => convolution_53
#   input_147 => add_115, mul_161, mul_162, sub_53
#   input_148 => clamp_max_34, clamp_min_34
# Graph fragment:
#   %convolution_53 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_33, %primals_324, %primals_325, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_53 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_53, %unsqueeze_425), kwargs = {})
#   %mul_161 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_53, %unsqueeze_427), kwargs = {})
#   %mul_162 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_161, %unsqueeze_429), kwargs = {})
#   %add_115 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_162, %unsqueeze_431), kwargs = {})
#   %clamp_min_34 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_115, 0.0), kwargs = {})
#   %clamp_max_34 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_34, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 47616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 744)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dc/cdcvuv4p6ekebtzdppe2nmdxvcbaggy52t4xvpukqqw4ru2sbvkb.py
# Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_156 => convolution_57
#   input_157 => add_124, mul_173, mul_174, sub_57
#   input_158 => clamp_max_36, clamp_min_36
# Graph fragment:
#   %convolution_57 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_122, %primals_348, %primals_349, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_57 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_57, %unsqueeze_457), kwargs = {})
#   %mul_173 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_57, %unsqueeze_459), kwargs = {})
#   %mul_174 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_173, %unsqueeze_461), kwargs = {})
#   %add_124 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_174, %unsqueeze_463), kwargs = {})
#   %clamp_min_36 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_124, 0.0), kwargs = {})
#   %clamp_max_36 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_36, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_45 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_45', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_45', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_45(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 141)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g2/cg22kbs5eidx45w2qtcw4kt4gfgem67okmnvnfkgtji35rdnk5ug.py
# Topologically Sorted Source Nodes: [input_159, input_160, input_161], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_159 => convolution_58
#   input_160 => add_126, mul_176, mul_177, sub_58
#   input_161 => clamp_max_37, clamp_min_37
# Graph fragment:
#   %convolution_58 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_36, %primals_354, %primals_355, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_58, %unsqueeze_465), kwargs = {})
#   %mul_176 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_58, %unsqueeze_467), kwargs = {})
#   %mul_177 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_176, %unsqueeze_469), kwargs = {})
#   %add_126 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_177, %unsqueeze_471), kwargs = {})
#   %clamp_min_37 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_126, 0.0), kwargs = {})
#   %clamp_max_37 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_37, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_46 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_46', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_46', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_46(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 54144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 846)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qc/cqcmv5wxmfsdgn3nyoicf556wckd3svxhqn4vsbwhyn5sfxgit52.py
# Topologically Sorted Source Nodes: [input_169, input_170, input_171], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_169 => convolution_62
#   input_170 => add_135, mul_188, mul_189, sub_62
#   input_171 => clamp_max_39, clamp_min_39
# Graph fragment:
#   %convolution_62 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_133, %primals_378, %primals_379, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_62 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_62, %unsqueeze_497), kwargs = {})
#   %mul_188 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_62, %unsqueeze_499), kwargs = {})
#   %mul_189 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_188, %unsqueeze_501), kwargs = {})
#   %add_135 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_189, %unsqueeze_503), kwargs = {})
#   %clamp_min_39 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_135, 0.0), kwargs = {})
#   %clamp_max_39 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_39, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_47 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_47', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_47', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 140)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uq/cuqv7bidwxwcmgufje6iok3fmcrg7mfbp3xag2cmexwlu5vhoqb3.py
# Topologically Sorted Source Nodes: [input_172, input_173, input_174], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_172 => convolution_63
#   input_173 => add_137, mul_191, mul_192, sub_63
#   input_174 => clamp_max_40, clamp_min_40
# Graph fragment:
#   %convolution_63 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_39, %primals_384, %primals_385, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_63 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_63, %unsqueeze_505), kwargs = {})
#   %mul_191 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_63, %unsqueeze_507), kwargs = {})
#   %mul_192 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_191, %unsqueeze_509), kwargs = {})
#   %add_137 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_192, %unsqueeze_511), kwargs = {})
#   %clamp_min_40 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_137, 0.0), kwargs = {})
#   %clamp_max_40 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_40, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 53760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 840)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kn/ckn7ofngrmxskqzyrq6asypfypg46srqddh2ljl37kiqcd2mlufw.py
# Topologically Sorted Source Nodes: [input_182, input_183, input_184], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_182 => convolution_67
#   input_183 => add_146, mul_203, mul_204, sub_67
#   input_184 => clamp_max_42, clamp_min_42
# Graph fragment:
#   %convolution_67 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_144, %primals_408, %primals_409, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_67 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_67, %unsqueeze_537), kwargs = {})
#   %mul_203 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_67, %unsqueeze_539), kwargs = {})
#   %mul_204 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_203, %unsqueeze_541), kwargs = {})
#   %add_146 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_204, %unsqueeze_543), kwargs = {})
#   %clamp_min_42 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_146, 0.0), kwargs = {})
#   %clamp_max_42 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_42, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_49 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_49', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_49', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_49(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 137)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6m/c6m22uho3hdjpfhlbwyyi7gyz4t2gy7cbjekhyvbbxyznfce3zi2.py
# Topologically Sorted Source Nodes: [input_185, input_186, input_187], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_185 => convolution_68
#   input_186 => add_148, mul_206, mul_207, sub_68
#   input_187 => clamp_max_43, clamp_min_43
# Graph fragment:
#   %convolution_68 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_42, %primals_414, %primals_415, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_68 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_545), kwargs = {})
#   %mul_206 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_68, %unsqueeze_547), kwargs = {})
#   %mul_207 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_206, %unsqueeze_549), kwargs = {})
#   %add_148 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_207, %unsqueeze_551), kwargs = {})
#   %clamp_min_43 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_148, 0.0), kwargs = {})
#   %clamp_max_43 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_43, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_50 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_50', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_50', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 822)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h5/ch5gd4bzdogtajezolpnrfygwxh2dyknqvhvlz5neyecmj7p24fd.py
# Topologically Sorted Source Nodes: [input_195, input_196, input_197], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_195 => convolution_72
#   input_196 => add_157, mul_218, mul_219, sub_72
#   input_197 => clamp_max_45, clamp_min_45
# Graph fragment:
#   %convolution_72 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_155, %primals_438, %primals_439, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_72 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_72, %unsqueeze_577), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_72, %unsqueeze_579), kwargs = {})
#   %mul_219 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_218, %unsqueeze_581), kwargs = {})
#   %add_157 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_219, %unsqueeze_583), kwargs = {})
#   %clamp_min_45 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_157, 0.0), kwargs = {})
#   %clamp_max_45 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_45, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_51 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_51', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_51', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 135)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cd/ccdbhgnwlief34io5atxxq3eizi44wk6av5nas2en732neyib6yh.py
# Topologically Sorted Source Nodes: [input_198, input_199, input_200], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_198 => convolution_73
#   input_199 => add_159, mul_221, mul_222, sub_73
#   input_200 => clamp_max_46, clamp_min_46
# Graph fragment:
#   %convolution_73 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_45, %primals_444, %primals_445, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_73 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_73, %unsqueeze_585), kwargs = {})
#   %mul_221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_73, %unsqueeze_587), kwargs = {})
#   %mul_222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_221, %unsqueeze_589), kwargs = {})
#   %add_159 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_222, %unsqueeze_591), kwargs = {})
#   %clamp_min_46 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_159, 0.0), kwargs = {})
#   %clamp_max_46 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_46, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_52 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_52', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 810)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mn/cmnnsdiznljwwhvgkm7vclkqpkq3yrqh5rju4kcep76grv4wnj5d.py
# Topologically Sorted Source Nodes: [input_208, input_209, input_210], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_208 => convolution_77
#   input_209 => add_168, mul_233, mul_234, sub_77
#   input_210 => clamp_max_48, clamp_min_48
# Graph fragment:
#   %convolution_77 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_166, %primals_468, %primals_469, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_77 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_77, %unsqueeze_617), kwargs = {})
#   %mul_233 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_77, %unsqueeze_619), kwargs = {})
#   %mul_234 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_233, %unsqueeze_621), kwargs = {})
#   %add_168 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_234, %unsqueeze_623), kwargs = {})
#   %clamp_min_48 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_168, 0.0), kwargs = {})
#   %clamp_max_48 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_48, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_53 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_53', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_53', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_53(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 133)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yt/cytyiifrfqymy5evozeiz4hepi46nmofiaqsxchg2dbcg2laicxq.py
# Topologically Sorted Source Nodes: [input_211, input_212, input_213], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_211 => convolution_78
#   input_212 => add_170, mul_236, mul_237, sub_78
#   input_213 => clamp_max_49, clamp_min_49
# Graph fragment:
#   %convolution_78 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_48, %primals_474, %primals_475, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_78 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_78, %unsqueeze_625), kwargs = {})
#   %mul_236 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_78, %unsqueeze_627), kwargs = {})
#   %mul_237 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_236, %unsqueeze_629), kwargs = {})
#   %add_170 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_237, %unsqueeze_631), kwargs = {})
#   %clamp_min_49 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_170, 0.0), kwargs = {})
#   %clamp_max_49 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_49, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_54 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_54', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_54', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 798)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yr/cyradsc7tbsmv5la5z6kjppk47ngfxwql7h3l74pec4bsjglezqy.py
# Topologically Sorted Source Nodes: [input_234, input_235, input_236], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_234 => convolution_87
#   input_235 => add_190, mul_263, mul_264, sub_87
#   input_236 => clamp_max_54, clamp_min_54
# Graph fragment:
#   %convolution_87 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_188, %primals_528, %primals_529, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_87 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_87, %unsqueeze_697), kwargs = {})
#   %mul_263 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_87, %unsqueeze_699), kwargs = {})
#   %mul_264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_263, %unsqueeze_701), kwargs = {})
#   %add_190 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_264, %unsqueeze_703), kwargs = {})
#   %clamp_min_54 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_190, 0.0), kwargs = {})
#   %clamp_max_54 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_54, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_55 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_55', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_55', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_55(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 124800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1950)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/22/c22mmdzyd5ncduwtsapz6hkm52spy5r6p2hotx5znzkhfq76absq.py
# Topologically Sorted Source Nodes: [input_237, input_238, input_239], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_237 => convolution_88
#   input_238 => add_192, mul_266, mul_267, sub_88
#   input_239 => clamp_max_55, clamp_min_55
# Graph fragment:
#   %convolution_88 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_54, %primals_534, %primals_535, [2, 2], [1, 1], [1, 1], False, [0, 0], 1950), kwargs = {})
#   %sub_88 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_88, %unsqueeze_705), kwargs = {})
#   %mul_266 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_88, %unsqueeze_707), kwargs = {})
#   %mul_267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_266, %unsqueeze_709), kwargs = {})
#   %add_192 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_267, %unsqueeze_711), kwargs = {})
#   %clamp_min_55 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_192, 0.0), kwargs = {})
#   %clamp_max_55 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_55, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_56 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_56', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_56', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1950)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ol/colg2q3vn2v4qolztyf5rlucfj3o62kq4y2lggoqh24ccujcvmcc.py
# Topologically Sorted Source Nodes: [input_240, input_241], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_240 => convolution_89
#   input_241 => add_194, mul_269, mul_270, sub_89
# Graph fragment:
#   %convolution_89 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_55, %primals_540, %primals_541, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_89 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_89, %unsqueeze_713), kwargs = {})
#   %mul_269 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_89, %unsqueeze_715), kwargs = {})
#   %mul_270 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_269, %unsqueeze_717), kwargs = {})
#   %add_194 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_270, %unsqueeze_719), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_57 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_57', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_57', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_57(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 545)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qx/cqxuplzai6jadnrehxwgcwduuq5hshmwhycddtjxip7m7fsvufn6.py
# Topologically Sorted Source Nodes: [input_242, input_243, input_244], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_242 => convolution_90
#   input_243 => add_196, mul_272, mul_273, sub_90
#   input_244 => clamp_max_56, clamp_min_56
# Graph fragment:
#   %convolution_90 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_194, %primals_546, %primals_547, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_90 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_90, %unsqueeze_721), kwargs = {})
#   %mul_272 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_90, %unsqueeze_723), kwargs = {})
#   %mul_273 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_272, %unsqueeze_725), kwargs = {})
#   %add_196 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_273, %unsqueeze_727), kwargs = {})
#   %clamp_min_56 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_196, 0.0), kwargs = {})
#   %clamp_max_56 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_56, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_58 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_58', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 276)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z5/cz5pjdppwiflmsaibii425qqwxcig5veneyiqyvygdywa4jzlopy.py
# Topologically Sorted Source Nodes: [input_245, input_246, input_247], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_245 => convolution_91
#   input_246 => add_198, mul_275, mul_276, sub_91
#   input_247 => clamp_max_57, clamp_min_57
# Graph fragment:
#   %convolution_91 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_56, %primals_552, %primals_553, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_91 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_91, %unsqueeze_729), kwargs = {})
#   %mul_275 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_91, %unsqueeze_731), kwargs = {})
#   %mul_276 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_275, %unsqueeze_733), kwargs = {})
#   %add_198 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_276, %unsqueeze_735), kwargs = {})
#   %clamp_min_57 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_198, 0.0), kwargs = {})
#   %clamp_max_57 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_57, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_59 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_59', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_59', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 26496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1656)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cv/ccvp4brgq6genrlnw3yf2qmwmdtwa4qg72k5m3x3g76drphg6yyn.py
# Topologically Sorted Source Nodes: [input_251, input_252, input_253, input_254, out_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_251 => convolution_93
#   input_252 => add_202, mul_281, mul_282, sub_93
#   input_253 => convolution_94
#   input_254 => add_204, mul_284, mul_285, sub_94
#   out_15 => add_205
# Graph fragment:
#   %convolution_93 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_58, %primals_564, %primals_565, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_93 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_93, %unsqueeze_745), kwargs = {})
#   %mul_281 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_93, %unsqueeze_747), kwargs = {})
#   %mul_282 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_281, %unsqueeze_749), kwargs = {})
#   %add_202 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_282, %unsqueeze_751), kwargs = {})
#   %convolution_94 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_194, %primals_570, %primals_571, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_94, %unsqueeze_753), kwargs = {})
#   %mul_284 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %unsqueeze_755), kwargs = {})
#   %mul_285 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_284, %unsqueeze_757), kwargs = {})
#   %add_204 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_285, %unsqueeze_759), kwargs = {})
#   %add_205 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_202, %add_204), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_60 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_60', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_60', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_60(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 545)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ko/ckouol72gykdi2efx2xmukw4pqghirr6youhnxlhim5ptvojnida.py
# Topologically Sorted Source Nodes: [input_255, input_256, input_257], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_255 => convolution_95
#   input_256 => add_207, mul_287, mul_288, sub_95
#   input_257 => clamp_max_59, clamp_min_59
# Graph fragment:
#   %convolution_95 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_205, %primals_576, %primals_577, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_95, %unsqueeze_761), kwargs = {})
#   %mul_287 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %unsqueeze_763), kwargs = {})
#   %mul_288 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_287, %unsqueeze_765), kwargs = {})
#   %add_207 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_288, %unsqueeze_767), kwargs = {})
#   %clamp_min_59 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_207, 0.0), kwargs = {})
#   %clamp_max_59 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_59, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_61 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_61', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_61', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_61(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 230)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dc/cdcsqmyoxk7f6dosm56inrrl2crtohv6vokkzyctxpbe7jcqt254.py
# Topologically Sorted Source Nodes: [input_258, input_259, input_260], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_258 => convolution_96
#   input_259 => add_209, mul_290, mul_291, sub_96
#   input_260 => clamp_max_60, clamp_min_60
# Graph fragment:
#   %convolution_96 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_59, %primals_582, %primals_583, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_96 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_96, %unsqueeze_769), kwargs = {})
#   %mul_290 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_96, %unsqueeze_771), kwargs = {})
#   %mul_291 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_290, %unsqueeze_773), kwargs = {})
#   %add_209 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_291, %unsqueeze_775), kwargs = {})
#   %clamp_min_60 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_209, 0.0), kwargs = {})
#   %clamp_max_60 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_60, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_62 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_62', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_62', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_62(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1380)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xn/cxnoz3yakqouwsitcogjurdcvhiugq2xj3oedotwa74qcobwq4aa.py
# Topologically Sorted Source Nodes: [input_264, input_265, input_266, input_267, out_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_264 => convolution_98
#   input_265 => add_213, mul_296, mul_297, sub_98
#   input_266 => convolution_99
#   input_267 => add_215, mul_299, mul_300, sub_99
#   out_16 => add_216
# Graph fragment:
#   %convolution_98 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_61, %primals_594, %primals_595, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_98 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_98, %unsqueeze_785), kwargs = {})
#   %mul_296 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_98, %unsqueeze_787), kwargs = {})
#   %mul_297 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_296, %unsqueeze_789), kwargs = {})
#   %add_213 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_297, %unsqueeze_791), kwargs = {})
#   %convolution_99 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_59, %primals_600, %primals_601, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_99 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_99, %unsqueeze_793), kwargs = {})
#   %mul_299 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_99, %unsqueeze_795), kwargs = {})
#   %mul_300 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_299, %unsqueeze_797), kwargs = {})
#   %add_215 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_300, %unsqueeze_799), kwargs = {})
#   %add_216 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_213, %add_215), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_63 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_63', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_63', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_63(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 489)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2c/c2ckhluc2egv6m5cc7qdbwez5rmuvnx535jgazgecz6oixdofs2f.py
# Topologically Sorted Source Nodes: [input_268, input_269, input_270], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_268 => convolution_100
#   input_269 => add_218, mul_302, mul_303, sub_100
#   input_270 => clamp_max_62, clamp_min_62
# Graph fragment:
#   %convolution_100 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_216, %primals_606, %primals_607, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_100 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_100, %unsqueeze_801), kwargs = {})
#   %mul_302 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_100, %unsqueeze_803), kwargs = {})
#   %mul_303 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_302, %unsqueeze_805), kwargs = {})
#   %add_218 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_303, %unsqueeze_807), kwargs = {})
#   %clamp_min_62 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_218, 0.0), kwargs = {})
#   %clamp_max_62 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_62, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_64 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_64', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_64', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_64(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 213)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5d/c5dvo5lx5x332wc22cnvbhggdgf7p5op2hd4vttqvhat3dlkn75r.py
# Topologically Sorted Source Nodes: [input_271, input_272, input_273], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_271 => convolution_101
#   input_272 => add_220, mul_305, mul_306, sub_101
#   input_273 => clamp_max_63, clamp_min_63
# Graph fragment:
#   %convolution_101 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_62, %primals_612, %primals_613, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_101 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_101, %unsqueeze_809), kwargs = {})
#   %mul_305 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_101, %unsqueeze_811), kwargs = {})
#   %mul_306 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_305, %unsqueeze_813), kwargs = {})
#   %add_220 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_306, %unsqueeze_815), kwargs = {})
#   %clamp_min_63 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_220, 0.0), kwargs = {})
#   %clamp_max_63 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_63, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_65 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_65', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_65', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_65(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 20448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1278)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/52/c52mamuhwjfnsntkbihasyxzpiosomhduwsd2dldk7tm75jvrzgq.py
# Topologically Sorted Source Nodes: [input_277, input_278, input_279, input_280, out_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_277 => convolution_103
#   input_278 => add_224, mul_311, mul_312, sub_103
#   input_279 => convolution_104
#   input_280 => add_226, mul_314, mul_315, sub_104
#   out_17 => add_227
# Graph fragment:
#   %convolution_103 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_64, %primals_624, %primals_625, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_103 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_103, %unsqueeze_825), kwargs = {})
#   %mul_311 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_103, %unsqueeze_827), kwargs = {})
#   %mul_312 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_311, %unsqueeze_829), kwargs = {})
#   %add_224 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_312, %unsqueeze_831), kwargs = {})
#   %convolution_104 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_216, %primals_630, %primals_631, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_104 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_104, %unsqueeze_833), kwargs = {})
#   %mul_314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_104, %unsqueeze_835), kwargs = {})
#   %mul_315 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_314, %unsqueeze_837), kwargs = {})
#   %add_226 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_315, %unsqueeze_839), kwargs = {})
#   %add_227 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_224, %add_226), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_66 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_66', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_66', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_66(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 469)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m5/cm5qdosgcp2nwfcb3gk2jpcxmokiqdgduml2og4i5ycove65sxcj.py
# Topologically Sorted Source Nodes: [input_281, input_282, input_283], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_281 => convolution_105
#   input_282 => add_229, mul_317, mul_318, sub_105
#   input_283 => clamp_max_65, clamp_min_65
# Graph fragment:
#   %convolution_105 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_227, %primals_636, %primals_637, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_105 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_105, %unsqueeze_841), kwargs = {})
#   %mul_317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_105, %unsqueeze_843), kwargs = {})
#   %mul_318 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_317, %unsqueeze_845), kwargs = {})
#   %add_229 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_318, %unsqueeze_847), kwargs = {})
#   %clamp_min_65 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_229, 0.0), kwargs = {})
#   %clamp_max_65 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_65, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_67 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_67', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_67', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_67(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 189)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lu/cluv6gqusc2x3o4qzdfuvcn3d53dtbuccg5jfbncjqtagrkx4v3a.py
# Topologically Sorted Source Nodes: [input_284, input_285, input_286], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_284 => convolution_106
#   input_285 => add_231, mul_320, mul_321, sub_106
#   input_286 => clamp_max_66, clamp_min_66
# Graph fragment:
#   %convolution_106 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_65, %primals_642, %primals_643, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_106 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_106, %unsqueeze_849), kwargs = {})
#   %mul_320 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_106, %unsqueeze_851), kwargs = {})
#   %mul_321 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_320, %unsqueeze_853), kwargs = {})
#   %add_231 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_321, %unsqueeze_855), kwargs = {})
#   %clamp_min_66 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_231, 0.0), kwargs = {})
#   %clamp_max_66 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_66, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_68 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_68', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_68(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 105)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yj/cyjkgmlimyoipafz5phdpdyfrzadba4rdyqlgwmybnrz47ezyefd.py
# Topologically Sorted Source Nodes: [input_287], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_287 => _unsafe_index, _unsafe_index_1, add_234, mul_324, sub_108
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_66, [None, None, %convert_element_type_215, %convert_element_type_217]), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_66, [None, None, %convert_element_type_215, %clamp_max_68]), kwargs = {})
#   %sub_108 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_1, %_unsafe_index), kwargs = {})
#   %mul_324 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_108, %clamp_max_69), kwargs = {})
#   %add_234 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %mul_324), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_69 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_69', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_69', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_69(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 4) % 4)
    x0 = (xindex % 4)
    x2 = ((xindex // 16) % 105)
    x3 = xindex // 1680
    x5 = (xindex % 1680)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 2, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (x2 + 105*tmp8 + 210*tmp4 + 420*x3), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (x2 + 105*tmp13 + 210*tmp4 + 420*x3), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x5 + 1696*x3), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/4d/c4dkxjqtaemwfc4xs2zal55xmuvlllscevlwgspogeslnyrx7riu.py
# Topologically Sorted Source Nodes: [input_376, input_377, input_378], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_376 => convolution_140
#   input_377 => add_315, mul_432, mul_433, sub_150
#   input_378 => clamp_max_96, clamp_min_96
# Graph fragment:
#   %convolution_140 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_65, %primals_846, %primals_847, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_150 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_140, %unsqueeze_1121), kwargs = {})
#   %mul_432 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_150, %unsqueeze_1123), kwargs = {})
#   %mul_433 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_432, %unsqueeze_1125), kwargs = {})
#   %add_315 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_433, %unsqueeze_1127), kwargs = {})
#   %clamp_min_96 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_315, 0.0), kwargs = {})
#   %clamp_max_96 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_96, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_70 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_70', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_70', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_70(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1134)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/54/c544p6xuxs25w37kl4rfnlwiprcqzcqiad4rbxbswuhlqysrv6cv.py
# Topologically Sorted Source Nodes: [input_382, input_383, input_384, input_385, out_24], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_382 => convolution_142
#   input_383 => add_319, mul_438, mul_439, sub_152
#   input_384 => convolution_143
#   input_385 => add_321, mul_441, mul_442, sub_153
#   out_24 => add_322
# Graph fragment:
#   %convolution_142 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_97, %primals_858, %primals_859, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_152 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_142, %unsqueeze_1137), kwargs = {})
#   %mul_438 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_152, %unsqueeze_1139), kwargs = {})
#   %mul_439 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_438, %unsqueeze_1141), kwargs = {})
#   %add_319 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_439, %unsqueeze_1143), kwargs = {})
#   %convolution_143 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_65, %primals_864, %primals_865, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_153 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_143, %unsqueeze_1145), kwargs = {})
#   %mul_441 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_153, %unsqueeze_1147), kwargs = {})
#   %mul_442 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_441, %unsqueeze_1149), kwargs = {})
#   %add_321 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_442, %unsqueeze_1151), kwargs = {})
#   %add_322 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_319, %add_321), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_71 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_71', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_71', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_71(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 462)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d6/cd66beja5xb5xfapkryx3frbmhm5j5t7ivflxlrgtznhzrhe52yp.py
# Topologically Sorted Source Nodes: [input_386], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_386 => convolution_144
# Graph fragment:
#   %convolution_144 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_322, %primals_870, %primals_871, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_72 = async_compile.triton('triton_poi_fused_convolution_72', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_72', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_72(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 75)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p4/cp4eii4lrrcyvxbunrdtov4ghtp4jpwkm4vgl4mr73jsiyujrzwg.py
# Topologically Sorted Source Nodes: [input_387, input_388], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_387 => add_324, mul_444, mul_445, sub_154
#   input_388 => clamp_max_98, clamp_min_98
# Graph fragment:
#   %sub_154 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_144, %unsqueeze_1153), kwargs = {})
#   %mul_444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_154, %unsqueeze_1155), kwargs = {})
#   %mul_445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_444, %unsqueeze_1157), kwargs = {})
#   %add_324 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_445, %unsqueeze_1159), kwargs = {})
#   %clamp_min_98 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_324, 0.0), kwargs = {})
#   %clamp_max_98 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_98, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_73 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_73', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_73', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_73(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 300
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 75)
    y1 = yindex // 75
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 75*x2 + 300*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2 + 4*y3), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ii/ciiwtxkk6zfqxy2stwf47fdadkcbxcv5nmkragonmzlrghmgpqde.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_188, %add_236], 1), kwargs = {})
triton_poi_fused_cat_74 = async_compile.triton('triton_poi_fused_cat_74', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_74', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_74(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 430)
    x4 = xindex // 430
    x3 = xindex // 6880
    x5 = ((xindex // 430) % 16)
    x2 = ((xindex // 1720) % 4)
    x1 = ((xindex // 430) % 4)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 325, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (325*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 430, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x5 + 16*((-325) + x0) + 1696*x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([XBLOCK], 2, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tmp15 = tl.load(in_ptr3 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp15 + tmp11
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr4 + (105*tmp18 + 210*tmp14 + 420*x3 + ((-325) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr5 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 + tmp11
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr4 + (105*tmp23 + 210*tmp14 + 420*x3 + ((-325) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 - tmp19
    tmp26 = tl.load(in_ptr6 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp19 + tmp27
    tmp29 = tmp28 - tmp9
    tmp30 = tl.load(in_ptr7 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp9 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp6, tmp32, tmp33)
    tmp35 = tl.where(tmp4, tmp5, tmp34)
    tl.store(out_ptr0 + (x6), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pc/cpc37pgwf3w2aqz4lxfctgyytihblhl4msa6t3dn7jlamn7vxszw.py
# Topologically Sorted Source Nodes: [input_288, input_289, input_290], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_288 => convolution_107
#   input_289 => add_238, mul_328, mul_329, sub_112
#   input_290 => clamp_max_71, clamp_min_71
# Graph fragment:
#   %convolution_107 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %primals_648, %primals_649, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_112 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_107, %unsqueeze_857), kwargs = {})
#   %mul_328 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_112, %unsqueeze_859), kwargs = {})
#   %mul_329 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_328, %unsqueeze_861), kwargs = {})
#   %add_238 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_329, %unsqueeze_863), kwargs = {})
#   %clamp_min_71 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_238, 0.0), kwargs = {})
#   %clamp_max_71 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_71, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_75 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_75', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_75', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_75(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 113)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ds/cds5frgqcwtilc3jjtnnemfndn7ptzdsgmq3qwqq5xizfxctapzj.py
# Topologically Sorted Source Nodes: [input_291, input_292, input_293], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_291 => convolution_108
#   input_292 => add_240, mul_331, mul_332, sub_113
#   input_293 => clamp_max_72, clamp_min_72
# Graph fragment:
#   %convolution_108 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_71, %primals_654, %primals_655, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_113 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_108, %unsqueeze_865), kwargs = {})
#   %mul_331 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_113, %unsqueeze_867), kwargs = {})
#   %mul_332 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_331, %unsqueeze_869), kwargs = {})
#   %add_240 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_332, %unsqueeze_871), kwargs = {})
#   %clamp_min_72 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_240, 0.0), kwargs = {})
#   %clamp_max_72 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_72, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_76 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_76', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_76', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_76(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 678)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ax/cax6nmuz5niatetvhftwnsgm6aldzdh6doeqyuyigf5smfprg2iy.py
# Topologically Sorted Source Nodes: [input_301, input_302, input_303], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_301 => convolution_112
#   input_302 => add_249, mul_343, mul_344, sub_117
#   input_303 => clamp_max_74, clamp_min_74
# Graph fragment:
#   %convolution_112 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_247, %primals_678, %primals_679, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_117 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_112, %unsqueeze_897), kwargs = {})
#   %mul_343 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_117, %unsqueeze_899), kwargs = {})
#   %mul_344 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_343, %unsqueeze_901), kwargs = {})
#   %add_249 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_344, %unsqueeze_903), kwargs = {})
#   %clamp_min_74 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_249, 0.0), kwargs = {})
#   %clamp_max_74 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_74, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_77 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_77', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_77', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_77(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 99)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ew/cewswsfsxxjbt57p2lg7bu7ucbbwoiuibt75zh6oitmkm3twhnqi.py
# Topologically Sorted Source Nodes: [input_304, input_305, input_306], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_304 => convolution_113
#   input_305 => add_251, mul_346, mul_347, sub_118
#   input_306 => clamp_max_75, clamp_min_75
# Graph fragment:
#   %convolution_113 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_74, %primals_684, %primals_685, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_118 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_113, %unsqueeze_905), kwargs = {})
#   %mul_346 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_118, %unsqueeze_907), kwargs = {})
#   %mul_347 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_346, %unsqueeze_909), kwargs = {})
#   %add_251 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_347, %unsqueeze_911), kwargs = {})
#   %clamp_min_75 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_251, 0.0), kwargs = {})
#   %clamp_max_75 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_75, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_78 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_78', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_78', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_78(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 38016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 594)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ti/ctit4g7fijendz4owyrws22gglrizuupmeyvhwi2vhjfo4wfdzwc.py
# Topologically Sorted Source Nodes: [input_310, input_311, input_312, input_313, out_19], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_310 => convolution_115
#   input_311 => add_255, mul_352, mul_353, sub_120
#   input_312 => convolution_116
#   input_313 => add_257, mul_355, mul_356, sub_121
#   out_19 => add_258
# Graph fragment:
#   %convolution_115 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_76, %primals_696, %primals_697, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_120 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_115, %unsqueeze_921), kwargs = {})
#   %mul_352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_120, %unsqueeze_923), kwargs = {})
#   %mul_353 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_352, %unsqueeze_925), kwargs = {})
#   %add_255 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_353, %unsqueeze_927), kwargs = {})
#   %convolution_116 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_247, %primals_702, %primals_703, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_121 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_116, %unsqueeze_929), kwargs = {})
#   %mul_355 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_121, %unsqueeze_931), kwargs = {})
#   %mul_356 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_355, %unsqueeze_933), kwargs = {})
#   %add_257 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_356, %unsqueeze_935), kwargs = {})
#   %add_258 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_255, %add_257), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_79 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_79', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_79', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_79(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 207)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zx/czxomz7q5uio6natwy33ulx7uji5632inal6az272epmxr63dgsd.py
# Topologically Sorted Source Nodes: [input_314, input_315, input_316], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_314 => convolution_117
#   input_315 => add_260, mul_358, mul_359, sub_122
#   input_316 => clamp_max_77, clamp_min_77
# Graph fragment:
#   %convolution_117 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_258, %primals_708, %primals_709, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_117, %unsqueeze_937), kwargs = {})
#   %mul_358 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_939), kwargs = {})
#   %mul_359 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_358, %unsqueeze_941), kwargs = {})
#   %add_260 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_359, %unsqueeze_943), kwargs = {})
#   %clamp_min_77 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_260, 0.0), kwargs = {})
#   %clamp_max_77 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_77, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_80 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_80', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_80', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_80(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 98)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/in/cini2v5okityaa4u4sz7ld34azky5iws7kgdhsqvmumt6bppw3bo.py
# Topologically Sorted Source Nodes: [input_317, input_318, input_319], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_317 => convolution_118
#   input_318 => add_262, mul_361, mul_362, sub_123
#   input_319 => clamp_max_78, clamp_min_78
# Graph fragment:
#   %convolution_118 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_77, %primals_714, %primals_715, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_123 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_118, %unsqueeze_945), kwargs = {})
#   %mul_361 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_123, %unsqueeze_947), kwargs = {})
#   %mul_362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_361, %unsqueeze_949), kwargs = {})
#   %add_262 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_362, %unsqueeze_951), kwargs = {})
#   %clamp_min_78 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_262, 0.0), kwargs = {})
#   %clamp_max_78 : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_78, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_81 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_81', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_81', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_81(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 47)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5r/c5rjrr5dcvfdzrw6ummvz3b5x2vvvenux3e6jtrlwpa4fzt2ogrw.py
# Topologically Sorted Source Nodes: [input_320], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   input_320 => _unsafe_index_4, _unsafe_index_5, add_265, mul_365, sub_125
# Graph fragment:
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_78, [None, None, %convert_element_type_243, %convert_element_type_245]), kwargs = {})
#   %_unsafe_index_5 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%clamp_max_78, [None, None, %convert_element_type_243, %clamp_max_80]), kwargs = {})
#   %sub_125 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%_unsafe_index_5, %_unsafe_index_4), kwargs = {})
#   %mul_365 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_125, %clamp_max_81), kwargs = {})
#   %add_265 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_4, %mul_365), kwargs = {})
triton_poi_fused__unsafe_index_add_mul_sub_82 = async_compile.triton('triton_poi_fused__unsafe_index_add_mul_sub_82', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_add_mul_sub_82', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_add_mul_sub_82(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x2 = ((xindex // 64) % 47)
    x3 = xindex // 3008
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp9 = tl.load(in_ptr2 + (x2 + 47*tmp8 + 188*tmp4 + 752*x3), xmask, eviction_policy='evict_last')
    tmp11 = tmp10 + tmp1
    tmp12 = tmp10 < 0
    tmp13 = tl.where(tmp12, tmp11, tmp10)
    tmp14 = tl.load(in_ptr2 + (x2 + 47*tmp13 + 188*tmp4 + 752*x3), xmask, eviction_policy='evict_last')
    tmp15 = tmp14 - tmp9
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tl.store(out_ptr0 + (x5), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rk/crkyykhfrtc4ovku2knj7s3lj5ullhv6oz67kp4spbvydv5hfbvk.py
# Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_94, %add_267], 1), kwargs = {})
triton_poi_fused_cat_83 = async_compile.triton('triton_poi_fused_cat_83', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*i64', 'in_ptr3': '*i64', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_83', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_83(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 197)
    x4 = xindex // 197
    x3 = xindex // 12608
    x5 = ((xindex // 197) % 64)
    x2 = ((xindex // 1576) % 8)
    x1 = ((xindex // 197) % 8)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 150, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (150*x4 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x5 + 64*((-150) + x0) + 3008*x3), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr2 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tl.full([XBLOCK], 4, tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tmp15 = tl.load(in_ptr3 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp15 + tmp11
    tmp17 = tmp15 < 0
    tmp18 = tl.where(tmp17, tmp16, tmp15)
    tmp19 = tl.load(in_ptr4 + (47*tmp18 + 188*tmp14 + 752*x3 + ((-150) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.load(in_ptr5 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp20 + tmp11
    tmp22 = tmp20 < 0
    tmp23 = tl.where(tmp22, tmp21, tmp20)
    tmp24 = tl.load(in_ptr4 + (47*tmp23 + 188*tmp14 + 752*x3 + ((-150) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 - tmp19
    tmp26 = tl.load(in_ptr6 + (x1), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp19 + tmp27
    tmp29 = tmp28 - tmp9
    tmp30 = tl.load(in_ptr7 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tmp9 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp6, tmp32, tmp33)
    tmp35 = tl.where(tmp4, tmp5, tmp34)
    tl.store(out_ptr0 + (x6), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/75/c75dn35z2ppkedevaxhuc4uvqifnzdxmxt3uoo6n5b6l7ef66rla.py
# Topologically Sorted Source Nodes: [input_321, input_322, input_323], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_321 => convolution_119
#   input_322 => add_269, mul_369, mul_370, sub_129
#   input_323 => clamp_max_83, clamp_min_83
# Graph fragment:
#   %convolution_119 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_720, %primals_721, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_129 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_119, %unsqueeze_953), kwargs = {})
#   %mul_369 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_129, %unsqueeze_955), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_369, %unsqueeze_957), kwargs = {})
#   %add_269 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_370, %unsqueeze_959), kwargs = {})
#   %clamp_min_83 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_269, 0.0), kwargs = {})
#   %clamp_max_83 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_83, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_84 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_84', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_84', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_84(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 58)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ii/ciiz4v3hb7cmrfk4rmxf5s6nla2qshtqf47xi3wbv4se44fo3rux.py
# Topologically Sorted Source Nodes: [input_363, input_364, input_365], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_363 => convolution_135
#   input_364 => add_304, mul_417, mul_418, sub_145
#   input_365 => clamp_max_93, clamp_min_93
# Graph fragment:
#   %convolution_135 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_77, %primals_816, %primals_817, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_145 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_135, %unsqueeze_1081), kwargs = {})
#   %mul_417 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_145, %unsqueeze_1083), kwargs = {})
#   %mul_418 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_417, %unsqueeze_1085), kwargs = {})
#   %add_304 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_418, %unsqueeze_1087), kwargs = {})
#   %clamp_min_93 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_304, 0.0), kwargs = {})
#   %clamp_max_93 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_93, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_85 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_85', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_85', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_85(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 37632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 588)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5iokxricenmcw62dgwx6fld4ufynm3wx6xt4lzjnuldogs2ihly.py
# Topologically Sorted Source Nodes: [input_369, input_370, input_371, input_372, out_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_369 => convolution_137
#   input_370 => add_308, mul_423, mul_424, sub_147
#   input_371 => convolution_138
#   input_372 => add_310, mul_426, mul_427, sub_148
#   out_23 => add_311
# Graph fragment:
#   %convolution_137 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_94, %primals_828, %primals_829, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_147 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_137, %unsqueeze_1097), kwargs = {})
#   %mul_423 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_147, %unsqueeze_1099), kwargs = {})
#   %mul_424 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_423, %unsqueeze_1101), kwargs = {})
#   %add_308 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_424, %unsqueeze_1103), kwargs = {})
#   %convolution_138 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_77, %primals_834, %primals_835, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_148 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_138, %unsqueeze_1105), kwargs = {})
#   %mul_426 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_148, %unsqueeze_1107), kwargs = {})
#   %mul_427 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_426, %unsqueeze_1109), kwargs = {})
#   %add_310 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_427, %unsqueeze_1111), kwargs = {})
#   %add_311 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_308, %add_310), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_86 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_86', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_86', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_86(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 183)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rc/crc5pu3fdzz25bbv4ketmtebdbrhas6hl4d6bsv7rfh4buu5xx3s.py
# Topologically Sorted Source Nodes: [input_373], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_373 => convolution_139
# Graph fragment:
#   %convolution_139 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_311, %primals_840, %primals_841, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_87 = async_compile.triton('triton_poi_fused_convolution_87', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_87', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_87(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 75)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tb/ctbmu7w4uzhbrhwjcpetk3pooulbtcl2skdbpyjhfec7o3ntievy.py
# Topologically Sorted Source Nodes: [input_374, input_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_374 => add_313, mul_429, mul_430, sub_149
#   input_375 => clamp_max_95, clamp_min_95
# Graph fragment:
#   %sub_149 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_139, %unsqueeze_1113), kwargs = {})
#   %mul_429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_149, %unsqueeze_1115), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_429, %unsqueeze_1117), kwargs = {})
#   %add_313 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_430, %unsqueeze_1119), kwargs = {})
#   %clamp_min_95 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_313, 0.0), kwargs = {})
#   %clamp_max_95 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_95, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_88 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_88', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_88', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_88(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 300
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 75)
    y1 = yindex // 75
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 75*x2 + 1200*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2 + 16*y3), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/vz/cvz2q2oblbljnsvfj53zceol7ny2oij4ozide6wddxpcgju2qar4.py
# Topologically Sorted Source Nodes: [input_324, input_325, input_326], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_324 => convolution_120
#   input_325 => add_271, mul_372, mul_373, sub_130
#   input_326 => clamp_max_84, clamp_min_84
# Graph fragment:
#   %convolution_120 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_83, %primals_726, %primals_727, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_120, %unsqueeze_961), kwargs = {})
#   %mul_372 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_130, %unsqueeze_963), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_372, %unsqueeze_965), kwargs = {})
#   %add_271 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_373, %unsqueeze_967), kwargs = {})
#   %clamp_min_84 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_271, 0.0), kwargs = {})
#   %clamp_max_84 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_84, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_89 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_89', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_89', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_89(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 89088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 348)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5b/c5by446atbqr5lzagjjxrbtkdopdojtoychehixf3wrjpxwbjowt.py
# Topologically Sorted Source Nodes: [input_330, input_331, input_332, input_333, out_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_330 => convolution_122
#   input_331 => add_275, mul_378, mul_379, sub_132
#   input_332 => convolution_123
#   input_333 => add_277, mul_381, mul_382, sub_133
#   out_20 => add_278
# Graph fragment:
#   %convolution_122 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_85, %primals_738, %primals_739, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_132 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_122, %unsqueeze_977), kwargs = {})
#   %mul_378 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_132, %unsqueeze_979), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_378, %unsqueeze_981), kwargs = {})
#   %add_275 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_379, %unsqueeze_983), kwargs = {})
#   %convolution_123 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_744, %primals_745, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_133 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_123, %unsqueeze_985), kwargs = {})
#   %mul_381 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_133, %unsqueeze_987), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_381, %unsqueeze_989), kwargs = {})
#   %add_277 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_382, %unsqueeze_991), kwargs = {})
#   %add_278 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_275, %add_277), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_90 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_90', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_90', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_90(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 31232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 122)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qm/cqmiqlyhjj32ytdgzwk74y5xg2iexvodn7novgtm2s3365aoap6z.py
# Topologically Sorted Source Nodes: [input_334, input_335, input_336], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_334 => convolution_124
#   input_335 => add_280, mul_384, mul_385, sub_134
#   input_336 => clamp_max_86, clamp_min_86
# Graph fragment:
#   %convolution_124 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_278, %primals_750, %primals_751, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_134 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_993), kwargs = {})
#   %mul_384 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_134, %unsqueeze_995), kwargs = {})
#   %mul_385 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_384, %unsqueeze_997), kwargs = {})
#   %add_280 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_385, %unsqueeze_999), kwargs = {})
#   %clamp_min_86 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_280, 0.0), kwargs = {})
#   %clamp_max_86 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_86, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_91 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_91', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_91', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_91(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 52)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z2/cz26wksydbrpg57hkx66exyvrhs5wip5ivi7bqc774wpllzx3lqn.py
# Topologically Sorted Source Nodes: [input_337, input_338, input_339], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_337 => convolution_125
#   input_338 => add_282, mul_387, mul_388, sub_135
#   input_339 => clamp_max_87, clamp_min_87
# Graph fragment:
#   %convolution_125 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_86, %primals_756, %primals_757, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_135 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_1001), kwargs = {})
#   %mul_387 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_135, %unsqueeze_1003), kwargs = {})
#   %mul_388 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_387, %unsqueeze_1005), kwargs = {})
#   %add_282 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_388, %unsqueeze_1007), kwargs = {})
#   %clamp_min_87 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_282, 0.0), kwargs = {})
#   %clamp_max_87 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_87, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_92 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_92', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_92', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_92(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 79872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 312)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgdgor5op23xqa7ga44netlzrvcgm3kclxezl6222ibb5mwp3n47.py
# Topologically Sorted Source Nodes: [input_343, input_344, input_345, input_346, out_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_343 => convolution_127
#   input_344 => add_286, mul_393, mul_394, sub_137
#   input_345 => convolution_128
#   input_346 => add_288, mul_396, mul_397, sub_138
#   out_21 => add_289
# Graph fragment:
#   %convolution_127 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_88, %primals_768, %primals_769, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_137 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1017), kwargs = {})
#   %mul_393 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_137, %unsqueeze_1019), kwargs = {})
#   %mul_394 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_393, %unsqueeze_1021), kwargs = {})
#   %add_286 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_394, %unsqueeze_1023), kwargs = {})
#   %convolution_128 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_278, %primals_774, %primals_775, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_138 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_1025), kwargs = {})
#   %mul_396 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_138, %unsqueeze_1027), kwargs = {})
#   %mul_397 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_396, %unsqueeze_1029), kwargs = {})
#   %add_288 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_397, %unsqueeze_1031), kwargs = {})
#   %add_289 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_286, %add_288), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_93 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_93', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_93', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_93(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 22272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 87)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sv/csvgtgk66h66y6waerqyjcdrynkmpfi4nu5nzve7xa5ugyk45fub.py
# Topologically Sorted Source Nodes: [input_347, input_348, input_349], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_347 => convolution_129
#   input_348 => add_291, mul_399, mul_400, sub_139
#   input_349 => clamp_max_89, clamp_min_89
# Graph fragment:
#   %convolution_129 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_289, %primals_780, %primals_781, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_139 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_129, %unsqueeze_1033), kwargs = {})
#   %mul_399 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_139, %unsqueeze_1035), kwargs = {})
#   %mul_400 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_399, %unsqueeze_1037), kwargs = {})
#   %add_291 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_400, %unsqueeze_1039), kwargs = {})
#   %clamp_min_89 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_291, 0.0), kwargs = {})
#   %clamp_max_89 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_89, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_94 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_94', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_94', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_94(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 47)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfxe7abm22mldl2jrvayxcc7camu4xcqanqyw3fah6275ggsl7w7.py
# Topologically Sorted Source Nodes: [input_350, input_351, input_352], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_350 => convolution_130
#   input_351 => add_293, mul_402, mul_403, sub_140
#   input_352 => clamp_max_90, clamp_min_90
# Graph fragment:
#   %convolution_130 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_89, %primals_786, %primals_787, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_140 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_130, %unsqueeze_1041), kwargs = {})
#   %mul_402 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_140, %unsqueeze_1043), kwargs = {})
#   %mul_403 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_402, %unsqueeze_1045), kwargs = {})
#   %add_293 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_403, %unsqueeze_1047), kwargs = {})
#   %clamp_min_90 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_293, 0.0), kwargs = {})
#   %clamp_max_90 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_90, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_95 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_95', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_95', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_95(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 72192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 282)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = 0.0
    tmp19 = triton_helpers.maximum(tmp17, tmp18)
    tmp20 = 6.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(out_ptr0 + (x2), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rt/crtdtrtg23ctdpcfkgreyaq5fukabkchfr5qs47pseuixaqedb25.py
# Topologically Sorted Source Nodes: [input_356, input_357, input_358, input_359, out_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_356 => convolution_132
#   input_357 => add_297, mul_408, mul_409, sub_142
#   input_358 => convolution_133
#   input_359 => add_299, mul_411, mul_412, sub_143
#   out_22 => add_300
# Graph fragment:
#   %convolution_132 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%clamp_max_91, %primals_798, %primals_799, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_142 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_132, %unsqueeze_1057), kwargs = {})
#   %mul_408 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_142, %unsqueeze_1059), kwargs = {})
#   %mul_409 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_408, %unsqueeze_1061), kwargs = {})
#   %add_297 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_409, %unsqueeze_1063), kwargs = {})
#   %convolution_133 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_289, %primals_804, %primals_805, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %sub_143 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_133, %unsqueeze_1065), kwargs = {})
#   %mul_411 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_143, %unsqueeze_1067), kwargs = {})
#   %mul_412 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_411, %unsqueeze_1069), kwargs = {})
#   %add_299 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_412, %unsqueeze_1071), kwargs = {})
#   %add_300 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_297, %add_299), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_96 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_96', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_96', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_96(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 23808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 93)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp7 = tmp2 - tmp6
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp18 = tmp16 * tmp17
    tmp20 = tmp18 + tmp19
    tmp22 = tmp5 - tmp21
    tmp24 = tmp23 + tmp9
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = tmp12 / tmp25
    tmp27 = tmp26 * tmp14
    tmp28 = tmp22 * tmp27
    tmp30 = tmp28 * tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp20 + tmp32
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
    tl.store(in_out_ptr1 + (x2), tmp5, xmask)
    tl.store(out_ptr0 + (x2), tmp33, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/hz/chz5mmryo4rd5abwpdzymihvwjwssxwxwiyvwlwpfxb6btfoi35i.py
# Topologically Sorted Source Nodes: [input_360], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_360 => convolution_134
# Graph fragment:
#   %convolution_134 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_300, %primals_810, %primals_811, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_97 = async_compile.triton('triton_poi_fused_convolution_97', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_97', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_97(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 75)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7h/c7ho2q47m5aek7vh3bpdgfi4qdenhn2dmvz6ower633fdyy7is6v.py
# Topologically Sorted Source Nodes: [input_361, input_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_361 => add_302, mul_414, mul_415, sub_144
#   input_362 => clamp_max_92, clamp_min_92
# Graph fragment:
#   %sub_144 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_134, %unsqueeze_1073), kwargs = {})
#   %mul_414 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_144, %unsqueeze_1075), kwargs = {})
#   %mul_415 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_414, %unsqueeze_1077), kwargs = {})
#   %add_302 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_415, %unsqueeze_1079), kwargs = {})
#   %clamp_min_92 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_302, 0.0), kwargs = {})
#   %clamp_max_92 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_92, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_98 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_98', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_98', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_98(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 300
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 75)
    y1 = yindex // 75
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 75*x2 + 4800*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2 + 64*y3), tmp19, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875 = args
    args.clear()
    assert_size_stride(primals_1, (12, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (12, ), (1, ))
    assert_size_stride(primals_3, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_4, (12, ), (1, ))
    assert_size_stride(primals_5, (12, ), (1, ))
    assert_size_stride(primals_6, (12, ), (1, ))
    assert_size_stride(primals_7, (12, ), (1, ))
    assert_size_stride(primals_8, (24, 12, 3, 3), (108, 9, 3, 1))
    assert_size_stride(primals_9, (24, ), (1, ))
    assert_size_stride(primals_10, (24, ), (1, ))
    assert_size_stride(primals_11, (24, ), (1, ))
    assert_size_stride(primals_12, (24, ), (1, ))
    assert_size_stride(primals_13, (24, ), (1, ))
    assert_size_stride(primals_14, (7, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_15, (7, ), (1, ))
    assert_size_stride(primals_16, (7, ), (1, ))
    assert_size_stride(primals_17, (7, ), (1, ))
    assert_size_stride(primals_18, (7, ), (1, ))
    assert_size_stride(primals_19, (7, ), (1, ))
    assert_size_stride(primals_20, (42, 7, 1, 1), (7, 1, 1, 1))
    assert_size_stride(primals_21, (42, ), (1, ))
    assert_size_stride(primals_22, (42, ), (1, ))
    assert_size_stride(primals_23, (42, ), (1, ))
    assert_size_stride(primals_24, (42, ), (1, ))
    assert_size_stride(primals_25, (42, ), (1, ))
    assert_size_stride(primals_26, (42, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_27, (42, ), (1, ))
    assert_size_stride(primals_28, (42, ), (1, ))
    assert_size_stride(primals_29, (42, ), (1, ))
    assert_size_stride(primals_30, (42, ), (1, ))
    assert_size_stride(primals_31, (42, ), (1, ))
    assert_size_stride(primals_32, (24, 42, 1, 1), (42, 1, 1, 1))
    assert_size_stride(primals_33, (24, ), (1, ))
    assert_size_stride(primals_34, (24, ), (1, ))
    assert_size_stride(primals_35, (24, ), (1, ))
    assert_size_stride(primals_36, (24, ), (1, ))
    assert_size_stride(primals_37, (24, ), (1, ))
    assert_size_stride(primals_38, (24, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_39, (24, ), (1, ))
    assert_size_stride(primals_40, (24, ), (1, ))
    assert_size_stride(primals_41, (24, ), (1, ))
    assert_size_stride(primals_42, (24, ), (1, ))
    assert_size_stride(primals_43, (24, ), (1, ))
    assert_size_stride(primals_44, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_45, (144, ), (1, ))
    assert_size_stride(primals_46, (144, ), (1, ))
    assert_size_stride(primals_47, (144, ), (1, ))
    assert_size_stride(primals_48, (144, ), (1, ))
    assert_size_stride(primals_49, (144, ), (1, ))
    assert_size_stride(primals_50, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_51, (144, ), (1, ))
    assert_size_stride(primals_52, (144, ), (1, ))
    assert_size_stride(primals_53, (144, ), (1, ))
    assert_size_stride(primals_54, (144, ), (1, ))
    assert_size_stride(primals_55, (144, ), (1, ))
    assert_size_stride(primals_56, (70, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_57, (70, ), (1, ))
    assert_size_stride(primals_58, (70, ), (1, ))
    assert_size_stride(primals_59, (70, ), (1, ))
    assert_size_stride(primals_60, (70, ), (1, ))
    assert_size_stride(primals_61, (70, ), (1, ))
    assert_size_stride(primals_62, (25, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_63, (25, ), (1, ))
    assert_size_stride(primals_64, (25, ), (1, ))
    assert_size_stride(primals_65, (25, ), (1, ))
    assert_size_stride(primals_66, (25, ), (1, ))
    assert_size_stride(primals_67, (25, ), (1, ))
    assert_size_stride(primals_68, (150, 25, 1, 1), (25, 1, 1, 1))
    assert_size_stride(primals_69, (150, ), (1, ))
    assert_size_stride(primals_70, (150, ), (1, ))
    assert_size_stride(primals_71, (150, ), (1, ))
    assert_size_stride(primals_72, (150, ), (1, ))
    assert_size_stride(primals_73, (150, ), (1, ))
    assert_size_stride(primals_74, (150, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_75, (150, ), (1, ))
    assert_size_stride(primals_76, (150, ), (1, ))
    assert_size_stride(primals_77, (150, ), (1, ))
    assert_size_stride(primals_78, (150, ), (1, ))
    assert_size_stride(primals_79, (150, ), (1, ))
    assert_size_stride(primals_80, (70, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_81, (70, ), (1, ))
    assert_size_stride(primals_82, (70, ), (1, ))
    assert_size_stride(primals_83, (70, ), (1, ))
    assert_size_stride(primals_84, (70, ), (1, ))
    assert_size_stride(primals_85, (70, ), (1, ))
    assert_size_stride(primals_86, (70, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_87, (70, ), (1, ))
    assert_size_stride(primals_88, (70, ), (1, ))
    assert_size_stride(primals_89, (70, ), (1, ))
    assert_size_stride(primals_90, (70, ), (1, ))
    assert_size_stride(primals_91, (70, ), (1, ))
    assert_size_stride(primals_92, (24, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_93, (24, ), (1, ))
    assert_size_stride(primals_94, (24, ), (1, ))
    assert_size_stride(primals_95, (24, ), (1, ))
    assert_size_stride(primals_96, (24, ), (1, ))
    assert_size_stride(primals_97, (24, ), (1, ))
    assert_size_stride(primals_98, (144, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(primals_99, (144, ), (1, ))
    assert_size_stride(primals_100, (144, ), (1, ))
    assert_size_stride(primals_101, (144, ), (1, ))
    assert_size_stride(primals_102, (144, ), (1, ))
    assert_size_stride(primals_103, (144, ), (1, ))
    assert_size_stride(primals_104, (144, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_105, (144, ), (1, ))
    assert_size_stride(primals_106, (144, ), (1, ))
    assert_size_stride(primals_107, (144, ), (1, ))
    assert_size_stride(primals_108, (144, ), (1, ))
    assert_size_stride(primals_109, (144, ), (1, ))
    assert_size_stride(primals_110, (70, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_111, (70, ), (1, ))
    assert_size_stride(primals_112, (70, ), (1, ))
    assert_size_stride(primals_113, (70, ), (1, ))
    assert_size_stride(primals_114, (70, ), (1, ))
    assert_size_stride(primals_115, (70, ), (1, ))
    assert_size_stride(primals_116, (70, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_117, (70, ), (1, ))
    assert_size_stride(primals_118, (70, ), (1, ))
    assert_size_stride(primals_119, (70, ), (1, ))
    assert_size_stride(primals_120, (70, ), (1, ))
    assert_size_stride(primals_121, (70, ), (1, ))
    assert_size_stride(primals_122, (420, 70, 1, 1), (70, 1, 1, 1))
    assert_size_stride(primals_123, (420, ), (1, ))
    assert_size_stride(primals_124, (420, ), (1, ))
    assert_size_stride(primals_125, (420, ), (1, ))
    assert_size_stride(primals_126, (420, ), (1, ))
    assert_size_stride(primals_127, (420, ), (1, ))
    assert_size_stride(primals_128, (420, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_129, (420, ), (1, ))
    assert_size_stride(primals_130, (420, ), (1, ))
    assert_size_stride(primals_131, (420, ), (1, ))
    assert_size_stride(primals_132, (420, ), (1, ))
    assert_size_stride(primals_133, (420, ), (1, ))
    assert_size_stride(primals_134, (150, 420, 1, 1), (420, 1, 1, 1))
    assert_size_stride(primals_135, (150, ), (1, ))
    assert_size_stride(primals_136, (150, ), (1, ))
    assert_size_stride(primals_137, (150, ), (1, ))
    assert_size_stride(primals_138, (150, ), (1, ))
    assert_size_stride(primals_139, (150, ), (1, ))
    assert_size_stride(primals_140, (56, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_141, (56, ), (1, ))
    assert_size_stride(primals_142, (56, ), (1, ))
    assert_size_stride(primals_143, (56, ), (1, ))
    assert_size_stride(primals_144, (56, ), (1, ))
    assert_size_stride(primals_145, (56, ), (1, ))
    assert_size_stride(primals_146, (336, 56, 1, 1), (56, 1, 1, 1))
    assert_size_stride(primals_147, (336, ), (1, ))
    assert_size_stride(primals_148, (336, ), (1, ))
    assert_size_stride(primals_149, (336, ), (1, ))
    assert_size_stride(primals_150, (336, ), (1, ))
    assert_size_stride(primals_151, (336, ), (1, ))
    assert_size_stride(primals_152, (336, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_153, (336, ), (1, ))
    assert_size_stride(primals_154, (336, ), (1, ))
    assert_size_stride(primals_155, (336, ), (1, ))
    assert_size_stride(primals_156, (336, ), (1, ))
    assert_size_stride(primals_157, (336, ), (1, ))
    assert_size_stride(primals_158, (150, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_159, (150, ), (1, ))
    assert_size_stride(primals_160, (150, ), (1, ))
    assert_size_stride(primals_161, (150, ), (1, ))
    assert_size_stride(primals_162, (150, ), (1, ))
    assert_size_stride(primals_163, (150, ), (1, ))
    assert_size_stride(primals_164, (150, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_165, (150, ), (1, ))
    assert_size_stride(primals_166, (150, ), (1, ))
    assert_size_stride(primals_167, (150, ), (1, ))
    assert_size_stride(primals_168, (150, ), (1, ))
    assert_size_stride(primals_169, (150, ), (1, ))
    assert_size_stride(primals_170, (150, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_171, (150, ), (1, ))
    assert_size_stride(primals_172, (150, ), (1, ))
    assert_size_stride(primals_173, (150, ), (1, ))
    assert_size_stride(primals_174, (150, ), (1, ))
    assert_size_stride(primals_175, (150, ), (1, ))
    assert_size_stride(primals_176, (18, 150), (150, 1))
    assert_size_stride(primals_177, (18, ), (1, ))
    assert_size_stride(primals_178, (150, 18), (18, 1))
    assert_size_stride(primals_179, (150, ), (1, ))
    assert_size_stride(primals_180, (73, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_181, (73, ), (1, ))
    assert_size_stride(primals_182, (73, ), (1, ))
    assert_size_stride(primals_183, (73, ), (1, ))
    assert_size_stride(primals_184, (73, ), (1, ))
    assert_size_stride(primals_185, (73, ), (1, ))
    assert_size_stride(primals_186, (438, 73, 1, 1), (73, 1, 1, 1))
    assert_size_stride(primals_187, (438, ), (1, ))
    assert_size_stride(primals_188, (438, ), (1, ))
    assert_size_stride(primals_189, (438, ), (1, ))
    assert_size_stride(primals_190, (438, ), (1, ))
    assert_size_stride(primals_191, (438, ), (1, ))
    assert_size_stride(primals_192, (438, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_193, (438, ), (1, ))
    assert_size_stride(primals_194, (438, ), (1, ))
    assert_size_stride(primals_195, (438, ), (1, ))
    assert_size_stride(primals_196, (438, ), (1, ))
    assert_size_stride(primals_197, (438, ), (1, ))
    assert_size_stride(primals_198, (150, 438, 1, 1), (438, 1, 1, 1))
    assert_size_stride(primals_199, (150, ), (1, ))
    assert_size_stride(primals_200, (150, ), (1, ))
    assert_size_stride(primals_201, (150, ), (1, ))
    assert_size_stride(primals_202, (150, ), (1, ))
    assert_size_stride(primals_203, (150, ), (1, ))
    assert_size_stride(primals_204, (150, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_205, (150, ), (1, ))
    assert_size_stride(primals_206, (150, ), (1, ))
    assert_size_stride(primals_207, (150, ), (1, ))
    assert_size_stride(primals_208, (150, ), (1, ))
    assert_size_stride(primals_209, (150, ), (1, ))
    assert_size_stride(primals_210, (71, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_211, (71, ), (1, ))
    assert_size_stride(primals_212, (71, ), (1, ))
    assert_size_stride(primals_213, (71, ), (1, ))
    assert_size_stride(primals_214, (71, ), (1, ))
    assert_size_stride(primals_215, (71, ), (1, ))
    assert_size_stride(primals_216, (426, 71, 1, 1), (71, 1, 1, 1))
    assert_size_stride(primals_217, (426, ), (1, ))
    assert_size_stride(primals_218, (426, ), (1, ))
    assert_size_stride(primals_219, (426, ), (1, ))
    assert_size_stride(primals_220, (426, ), (1, ))
    assert_size_stride(primals_221, (426, ), (1, ))
    assert_size_stride(primals_222, (426, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_223, (426, ), (1, ))
    assert_size_stride(primals_224, (426, ), (1, ))
    assert_size_stride(primals_225, (426, ), (1, ))
    assert_size_stride(primals_226, (426, ), (1, ))
    assert_size_stride(primals_227, (426, ), (1, ))
    assert_size_stride(primals_228, (150, 426, 1, 1), (426, 1, 1, 1))
    assert_size_stride(primals_229, (150, ), (1, ))
    assert_size_stride(primals_230, (150, ), (1, ))
    assert_size_stride(primals_231, (150, ), (1, ))
    assert_size_stride(primals_232, (150, ), (1, ))
    assert_size_stride(primals_233, (150, ), (1, ))
    assert_size_stride(primals_234, (150, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_235, (150, ), (1, ))
    assert_size_stride(primals_236, (150, ), (1, ))
    assert_size_stride(primals_237, (150, ), (1, ))
    assert_size_stride(primals_238, (150, ), (1, ))
    assert_size_stride(primals_239, (150, ), (1, ))
    assert_size_stride(primals_240, (75, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_241, (75, ), (1, ))
    assert_size_stride(primals_242, (75, ), (1, ))
    assert_size_stride(primals_243, (75, ), (1, ))
    assert_size_stride(primals_244, (75, ), (1, ))
    assert_size_stride(primals_245, (75, ), (1, ))
    assert_size_stride(primals_246, (450, 75, 1, 1), (75, 1, 1, 1))
    assert_size_stride(primals_247, (450, ), (1, ))
    assert_size_stride(primals_248, (450, ), (1, ))
    assert_size_stride(primals_249, (450, ), (1, ))
    assert_size_stride(primals_250, (450, ), (1, ))
    assert_size_stride(primals_251, (450, ), (1, ))
    assert_size_stride(primals_252, (450, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_253, (450, ), (1, ))
    assert_size_stride(primals_254, (450, ), (1, ))
    assert_size_stride(primals_255, (450, ), (1, ))
    assert_size_stride(primals_256, (450, ), (1, ))
    assert_size_stride(primals_257, (450, ), (1, ))
    assert_size_stride(primals_258, (150, 450, 1, 1), (450, 1, 1, 1))
    assert_size_stride(primals_259, (150, ), (1, ))
    assert_size_stride(primals_260, (150, ), (1, ))
    assert_size_stride(primals_261, (150, ), (1, ))
    assert_size_stride(primals_262, (150, ), (1, ))
    assert_size_stride(primals_263, (150, ), (1, ))
    assert_size_stride(primals_264, (150, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_265, (150, ), (1, ))
    assert_size_stride(primals_266, (150, ), (1, ))
    assert_size_stride(primals_267, (150, ), (1, ))
    assert_size_stride(primals_268, (150, ), (1, ))
    assert_size_stride(primals_269, (150, ), (1, ))
    assert_size_stride(primals_270, (900, 150, 1, 1), (150, 1, 1, 1))
    assert_size_stride(primals_271, (900, ), (1, ))
    assert_size_stride(primals_272, (900, ), (1, ))
    assert_size_stride(primals_273, (900, ), (1, ))
    assert_size_stride(primals_274, (900, ), (1, ))
    assert_size_stride(primals_275, (900, ), (1, ))
    assert_size_stride(primals_276, (900, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_277, (900, ), (1, ))
    assert_size_stride(primals_278, (900, ), (1, ))
    assert_size_stride(primals_279, (900, ), (1, ))
    assert_size_stride(primals_280, (900, ), (1, ))
    assert_size_stride(primals_281, (900, ), (1, ))
    assert_size_stride(primals_282, (325, 900, 1, 1), (900, 1, 1, 1))
    assert_size_stride(primals_283, (325, ), (1, ))
    assert_size_stride(primals_284, (325, ), (1, ))
    assert_size_stride(primals_285, (325, ), (1, ))
    assert_size_stride(primals_286, (325, ), (1, ))
    assert_size_stride(primals_287, (325, ), (1, ))
    assert_size_stride(primals_288, (132, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_289, (132, ), (1, ))
    assert_size_stride(primals_290, (132, ), (1, ))
    assert_size_stride(primals_291, (132, ), (1, ))
    assert_size_stride(primals_292, (132, ), (1, ))
    assert_size_stride(primals_293, (132, ), (1, ))
    assert_size_stride(primals_294, (792, 132, 1, 1), (132, 1, 1, 1))
    assert_size_stride(primals_295, (792, ), (1, ))
    assert_size_stride(primals_296, (792, ), (1, ))
    assert_size_stride(primals_297, (792, ), (1, ))
    assert_size_stride(primals_298, (792, ), (1, ))
    assert_size_stride(primals_299, (792, ), (1, ))
    assert_size_stride(primals_300, (792, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_301, (792, ), (1, ))
    assert_size_stride(primals_302, (792, ), (1, ))
    assert_size_stride(primals_303, (792, ), (1, ))
    assert_size_stride(primals_304, (792, ), (1, ))
    assert_size_stride(primals_305, (792, ), (1, ))
    assert_size_stride(primals_306, (325, 792, 1, 1), (792, 1, 1, 1))
    assert_size_stride(primals_307, (325, ), (1, ))
    assert_size_stride(primals_308, (325, ), (1, ))
    assert_size_stride(primals_309, (325, ), (1, ))
    assert_size_stride(primals_310, (325, ), (1, ))
    assert_size_stride(primals_311, (325, ), (1, ))
    assert_size_stride(primals_312, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_313, (325, ), (1, ))
    assert_size_stride(primals_314, (325, ), (1, ))
    assert_size_stride(primals_315, (325, ), (1, ))
    assert_size_stride(primals_316, (325, ), (1, ))
    assert_size_stride(primals_317, (325, ), (1, ))
    assert_size_stride(primals_318, (124, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_319, (124, ), (1, ))
    assert_size_stride(primals_320, (124, ), (1, ))
    assert_size_stride(primals_321, (124, ), (1, ))
    assert_size_stride(primals_322, (124, ), (1, ))
    assert_size_stride(primals_323, (124, ), (1, ))
    assert_size_stride(primals_324, (744, 124, 1, 1), (124, 1, 1, 1))
    assert_size_stride(primals_325, (744, ), (1, ))
    assert_size_stride(primals_326, (744, ), (1, ))
    assert_size_stride(primals_327, (744, ), (1, ))
    assert_size_stride(primals_328, (744, ), (1, ))
    assert_size_stride(primals_329, (744, ), (1, ))
    assert_size_stride(primals_330, (744, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_331, (744, ), (1, ))
    assert_size_stride(primals_332, (744, ), (1, ))
    assert_size_stride(primals_333, (744, ), (1, ))
    assert_size_stride(primals_334, (744, ), (1, ))
    assert_size_stride(primals_335, (744, ), (1, ))
    assert_size_stride(primals_336, (325, 744, 1, 1), (744, 1, 1, 1))
    assert_size_stride(primals_337, (325, ), (1, ))
    assert_size_stride(primals_338, (325, ), (1, ))
    assert_size_stride(primals_339, (325, ), (1, ))
    assert_size_stride(primals_340, (325, ), (1, ))
    assert_size_stride(primals_341, (325, ), (1, ))
    assert_size_stride(primals_342, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_343, (325, ), (1, ))
    assert_size_stride(primals_344, (325, ), (1, ))
    assert_size_stride(primals_345, (325, ), (1, ))
    assert_size_stride(primals_346, (325, ), (1, ))
    assert_size_stride(primals_347, (325, ), (1, ))
    assert_size_stride(primals_348, (141, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_349, (141, ), (1, ))
    assert_size_stride(primals_350, (141, ), (1, ))
    assert_size_stride(primals_351, (141, ), (1, ))
    assert_size_stride(primals_352, (141, ), (1, ))
    assert_size_stride(primals_353, (141, ), (1, ))
    assert_size_stride(primals_354, (846, 141, 1, 1), (141, 1, 1, 1))
    assert_size_stride(primals_355, (846, ), (1, ))
    assert_size_stride(primals_356, (846, ), (1, ))
    assert_size_stride(primals_357, (846, ), (1, ))
    assert_size_stride(primals_358, (846, ), (1, ))
    assert_size_stride(primals_359, (846, ), (1, ))
    assert_size_stride(primals_360, (846, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_361, (846, ), (1, ))
    assert_size_stride(primals_362, (846, ), (1, ))
    assert_size_stride(primals_363, (846, ), (1, ))
    assert_size_stride(primals_364, (846, ), (1, ))
    assert_size_stride(primals_365, (846, ), (1, ))
    assert_size_stride(primals_366, (325, 846, 1, 1), (846, 1, 1, 1))
    assert_size_stride(primals_367, (325, ), (1, ))
    assert_size_stride(primals_368, (325, ), (1, ))
    assert_size_stride(primals_369, (325, ), (1, ))
    assert_size_stride(primals_370, (325, ), (1, ))
    assert_size_stride(primals_371, (325, ), (1, ))
    assert_size_stride(primals_372, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_373, (325, ), (1, ))
    assert_size_stride(primals_374, (325, ), (1, ))
    assert_size_stride(primals_375, (325, ), (1, ))
    assert_size_stride(primals_376, (325, ), (1, ))
    assert_size_stride(primals_377, (325, ), (1, ))
    assert_size_stride(primals_378, (140, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_379, (140, ), (1, ))
    assert_size_stride(primals_380, (140, ), (1, ))
    assert_size_stride(primals_381, (140, ), (1, ))
    assert_size_stride(primals_382, (140, ), (1, ))
    assert_size_stride(primals_383, (140, ), (1, ))
    assert_size_stride(primals_384, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(primals_385, (840, ), (1, ))
    assert_size_stride(primals_386, (840, ), (1, ))
    assert_size_stride(primals_387, (840, ), (1, ))
    assert_size_stride(primals_388, (840, ), (1, ))
    assert_size_stride(primals_389, (840, ), (1, ))
    assert_size_stride(primals_390, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_391, (840, ), (1, ))
    assert_size_stride(primals_392, (840, ), (1, ))
    assert_size_stride(primals_393, (840, ), (1, ))
    assert_size_stride(primals_394, (840, ), (1, ))
    assert_size_stride(primals_395, (840, ), (1, ))
    assert_size_stride(primals_396, (325, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_397, (325, ), (1, ))
    assert_size_stride(primals_398, (325, ), (1, ))
    assert_size_stride(primals_399, (325, ), (1, ))
    assert_size_stride(primals_400, (325, ), (1, ))
    assert_size_stride(primals_401, (325, ), (1, ))
    assert_size_stride(primals_402, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_403, (325, ), (1, ))
    assert_size_stride(primals_404, (325, ), (1, ))
    assert_size_stride(primals_405, (325, ), (1, ))
    assert_size_stride(primals_406, (325, ), (1, ))
    assert_size_stride(primals_407, (325, ), (1, ))
    assert_size_stride(primals_408, (137, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_409, (137, ), (1, ))
    assert_size_stride(primals_410, (137, ), (1, ))
    assert_size_stride(primals_411, (137, ), (1, ))
    assert_size_stride(primals_412, (137, ), (1, ))
    assert_size_stride(primals_413, (137, ), (1, ))
    assert_size_stride(primals_414, (822, 137, 1, 1), (137, 1, 1, 1))
    assert_size_stride(primals_415, (822, ), (1, ))
    assert_size_stride(primals_416, (822, ), (1, ))
    assert_size_stride(primals_417, (822, ), (1, ))
    assert_size_stride(primals_418, (822, ), (1, ))
    assert_size_stride(primals_419, (822, ), (1, ))
    assert_size_stride(primals_420, (822, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_421, (822, ), (1, ))
    assert_size_stride(primals_422, (822, ), (1, ))
    assert_size_stride(primals_423, (822, ), (1, ))
    assert_size_stride(primals_424, (822, ), (1, ))
    assert_size_stride(primals_425, (822, ), (1, ))
    assert_size_stride(primals_426, (325, 822, 1, 1), (822, 1, 1, 1))
    assert_size_stride(primals_427, (325, ), (1, ))
    assert_size_stride(primals_428, (325, ), (1, ))
    assert_size_stride(primals_429, (325, ), (1, ))
    assert_size_stride(primals_430, (325, ), (1, ))
    assert_size_stride(primals_431, (325, ), (1, ))
    assert_size_stride(primals_432, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_433, (325, ), (1, ))
    assert_size_stride(primals_434, (325, ), (1, ))
    assert_size_stride(primals_435, (325, ), (1, ))
    assert_size_stride(primals_436, (325, ), (1, ))
    assert_size_stride(primals_437, (325, ), (1, ))
    assert_size_stride(primals_438, (135, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_439, (135, ), (1, ))
    assert_size_stride(primals_440, (135, ), (1, ))
    assert_size_stride(primals_441, (135, ), (1, ))
    assert_size_stride(primals_442, (135, ), (1, ))
    assert_size_stride(primals_443, (135, ), (1, ))
    assert_size_stride(primals_444, (810, 135, 1, 1), (135, 1, 1, 1))
    assert_size_stride(primals_445, (810, ), (1, ))
    assert_size_stride(primals_446, (810, ), (1, ))
    assert_size_stride(primals_447, (810, ), (1, ))
    assert_size_stride(primals_448, (810, ), (1, ))
    assert_size_stride(primals_449, (810, ), (1, ))
    assert_size_stride(primals_450, (810, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_451, (810, ), (1, ))
    assert_size_stride(primals_452, (810, ), (1, ))
    assert_size_stride(primals_453, (810, ), (1, ))
    assert_size_stride(primals_454, (810, ), (1, ))
    assert_size_stride(primals_455, (810, ), (1, ))
    assert_size_stride(primals_456, (325, 810, 1, 1), (810, 1, 1, 1))
    assert_size_stride(primals_457, (325, ), (1, ))
    assert_size_stride(primals_458, (325, ), (1, ))
    assert_size_stride(primals_459, (325, ), (1, ))
    assert_size_stride(primals_460, (325, ), (1, ))
    assert_size_stride(primals_461, (325, ), (1, ))
    assert_size_stride(primals_462, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_463, (325, ), (1, ))
    assert_size_stride(primals_464, (325, ), (1, ))
    assert_size_stride(primals_465, (325, ), (1, ))
    assert_size_stride(primals_466, (325, ), (1, ))
    assert_size_stride(primals_467, (325, ), (1, ))
    assert_size_stride(primals_468, (133, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_469, (133, ), (1, ))
    assert_size_stride(primals_470, (133, ), (1, ))
    assert_size_stride(primals_471, (133, ), (1, ))
    assert_size_stride(primals_472, (133, ), (1, ))
    assert_size_stride(primals_473, (133, ), (1, ))
    assert_size_stride(primals_474, (798, 133, 1, 1), (133, 1, 1, 1))
    assert_size_stride(primals_475, (798, ), (1, ))
    assert_size_stride(primals_476, (798, ), (1, ))
    assert_size_stride(primals_477, (798, ), (1, ))
    assert_size_stride(primals_478, (798, ), (1, ))
    assert_size_stride(primals_479, (798, ), (1, ))
    assert_size_stride(primals_480, (798, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_481, (798, ), (1, ))
    assert_size_stride(primals_482, (798, ), (1, ))
    assert_size_stride(primals_483, (798, ), (1, ))
    assert_size_stride(primals_484, (798, ), (1, ))
    assert_size_stride(primals_485, (798, ), (1, ))
    assert_size_stride(primals_486, (325, 798, 1, 1), (798, 1, 1, 1))
    assert_size_stride(primals_487, (325, ), (1, ))
    assert_size_stride(primals_488, (325, ), (1, ))
    assert_size_stride(primals_489, (325, ), (1, ))
    assert_size_stride(primals_490, (325, ), (1, ))
    assert_size_stride(primals_491, (325, ), (1, ))
    assert_size_stride(primals_492, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_493, (325, ), (1, ))
    assert_size_stride(primals_494, (325, ), (1, ))
    assert_size_stride(primals_495, (325, ), (1, ))
    assert_size_stride(primals_496, (325, ), (1, ))
    assert_size_stride(primals_497, (325, ), (1, ))
    assert_size_stride(primals_498, (140, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_499, (140, ), (1, ))
    assert_size_stride(primals_500, (140, ), (1, ))
    assert_size_stride(primals_501, (140, ), (1, ))
    assert_size_stride(primals_502, (140, ), (1, ))
    assert_size_stride(primals_503, (140, ), (1, ))
    assert_size_stride(primals_504, (840, 140, 1, 1), (140, 1, 1, 1))
    assert_size_stride(primals_505, (840, ), (1, ))
    assert_size_stride(primals_506, (840, ), (1, ))
    assert_size_stride(primals_507, (840, ), (1, ))
    assert_size_stride(primals_508, (840, ), (1, ))
    assert_size_stride(primals_509, (840, ), (1, ))
    assert_size_stride(primals_510, (840, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_511, (840, ), (1, ))
    assert_size_stride(primals_512, (840, ), (1, ))
    assert_size_stride(primals_513, (840, ), (1, ))
    assert_size_stride(primals_514, (840, ), (1, ))
    assert_size_stride(primals_515, (840, ), (1, ))
    assert_size_stride(primals_516, (325, 840, 1, 1), (840, 1, 1, 1))
    assert_size_stride(primals_517, (325, ), (1, ))
    assert_size_stride(primals_518, (325, ), (1, ))
    assert_size_stride(primals_519, (325, ), (1, ))
    assert_size_stride(primals_520, (325, ), (1, ))
    assert_size_stride(primals_521, (325, ), (1, ))
    assert_size_stride(primals_522, (325, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_523, (325, ), (1, ))
    assert_size_stride(primals_524, (325, ), (1, ))
    assert_size_stride(primals_525, (325, ), (1, ))
    assert_size_stride(primals_526, (325, ), (1, ))
    assert_size_stride(primals_527, (325, ), (1, ))
    assert_size_stride(primals_528, (1950, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_529, (1950, ), (1, ))
    assert_size_stride(primals_530, (1950, ), (1, ))
    assert_size_stride(primals_531, (1950, ), (1, ))
    assert_size_stride(primals_532, (1950, ), (1, ))
    assert_size_stride(primals_533, (1950, ), (1, ))
    assert_size_stride(primals_534, (1950, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_535, (1950, ), (1, ))
    assert_size_stride(primals_536, (1950, ), (1, ))
    assert_size_stride(primals_537, (1950, ), (1, ))
    assert_size_stride(primals_538, (1950, ), (1, ))
    assert_size_stride(primals_539, (1950, ), (1, ))
    assert_size_stride(primals_540, (545, 1950, 1, 1), (1950, 1, 1, 1))
    assert_size_stride(primals_541, (545, ), (1, ))
    assert_size_stride(primals_542, (545, ), (1, ))
    assert_size_stride(primals_543, (545, ), (1, ))
    assert_size_stride(primals_544, (545, ), (1, ))
    assert_size_stride(primals_545, (545, ), (1, ))
    assert_size_stride(primals_546, (276, 545, 1, 1), (545, 1, 1, 1))
    assert_size_stride(primals_547, (276, ), (1, ))
    assert_size_stride(primals_548, (276, ), (1, ))
    assert_size_stride(primals_549, (276, ), (1, ))
    assert_size_stride(primals_550, (276, ), (1, ))
    assert_size_stride(primals_551, (276, ), (1, ))
    assert_size_stride(primals_552, (1656, 276, 1, 1), (276, 1, 1, 1))
    assert_size_stride(primals_553, (1656, ), (1, ))
    assert_size_stride(primals_554, (1656, ), (1, ))
    assert_size_stride(primals_555, (1656, ), (1, ))
    assert_size_stride(primals_556, (1656, ), (1, ))
    assert_size_stride(primals_557, (1656, ), (1, ))
    assert_size_stride(primals_558, (1656, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_559, (1656, ), (1, ))
    assert_size_stride(primals_560, (1656, ), (1, ))
    assert_size_stride(primals_561, (1656, ), (1, ))
    assert_size_stride(primals_562, (1656, ), (1, ))
    assert_size_stride(primals_563, (1656, ), (1, ))
    assert_size_stride(primals_564, (545, 1656, 1, 1), (1656, 1, 1, 1))
    assert_size_stride(primals_565, (545, ), (1, ))
    assert_size_stride(primals_566, (545, ), (1, ))
    assert_size_stride(primals_567, (545, ), (1, ))
    assert_size_stride(primals_568, (545, ), (1, ))
    assert_size_stride(primals_569, (545, ), (1, ))
    assert_size_stride(primals_570, (545, 545, 1, 1), (545, 1, 1, 1))
    assert_size_stride(primals_571, (545, ), (1, ))
    assert_size_stride(primals_572, (545, ), (1, ))
    assert_size_stride(primals_573, (545, ), (1, ))
    assert_size_stride(primals_574, (545, ), (1, ))
    assert_size_stride(primals_575, (545, ), (1, ))
    assert_size_stride(primals_576, (230, 545, 1, 1), (545, 1, 1, 1))
    assert_size_stride(primals_577, (230, ), (1, ))
    assert_size_stride(primals_578, (230, ), (1, ))
    assert_size_stride(primals_579, (230, ), (1, ))
    assert_size_stride(primals_580, (230, ), (1, ))
    assert_size_stride(primals_581, (230, ), (1, ))
    assert_size_stride(primals_582, (1380, 230, 1, 1), (230, 1, 1, 1))
    assert_size_stride(primals_583, (1380, ), (1, ))
    assert_size_stride(primals_584, (1380, ), (1, ))
    assert_size_stride(primals_585, (1380, ), (1, ))
    assert_size_stride(primals_586, (1380, ), (1, ))
    assert_size_stride(primals_587, (1380, ), (1, ))
    assert_size_stride(primals_588, (1380, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_589, (1380, ), (1, ))
    assert_size_stride(primals_590, (1380, ), (1, ))
    assert_size_stride(primals_591, (1380, ), (1, ))
    assert_size_stride(primals_592, (1380, ), (1, ))
    assert_size_stride(primals_593, (1380, ), (1, ))
    assert_size_stride(primals_594, (489, 1380, 1, 1), (1380, 1, 1, 1))
    assert_size_stride(primals_595, (489, ), (1, ))
    assert_size_stride(primals_596, (489, ), (1, ))
    assert_size_stride(primals_597, (489, ), (1, ))
    assert_size_stride(primals_598, (489, ), (1, ))
    assert_size_stride(primals_599, (489, ), (1, ))
    assert_size_stride(primals_600, (489, 230, 1, 1), (230, 1, 1, 1))
    assert_size_stride(primals_601, (489, ), (1, ))
    assert_size_stride(primals_602, (489, ), (1, ))
    assert_size_stride(primals_603, (489, ), (1, ))
    assert_size_stride(primals_604, (489, ), (1, ))
    assert_size_stride(primals_605, (489, ), (1, ))
    assert_size_stride(primals_606, (213, 489, 1, 1), (489, 1, 1, 1))
    assert_size_stride(primals_607, (213, ), (1, ))
    assert_size_stride(primals_608, (213, ), (1, ))
    assert_size_stride(primals_609, (213, ), (1, ))
    assert_size_stride(primals_610, (213, ), (1, ))
    assert_size_stride(primals_611, (213, ), (1, ))
    assert_size_stride(primals_612, (1278, 213, 1, 1), (213, 1, 1, 1))
    assert_size_stride(primals_613, (1278, ), (1, ))
    assert_size_stride(primals_614, (1278, ), (1, ))
    assert_size_stride(primals_615, (1278, ), (1, ))
    assert_size_stride(primals_616, (1278, ), (1, ))
    assert_size_stride(primals_617, (1278, ), (1, ))
    assert_size_stride(primals_618, (1278, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_619, (1278, ), (1, ))
    assert_size_stride(primals_620, (1278, ), (1, ))
    assert_size_stride(primals_621, (1278, ), (1, ))
    assert_size_stride(primals_622, (1278, ), (1, ))
    assert_size_stride(primals_623, (1278, ), (1, ))
    assert_size_stride(primals_624, (469, 1278, 1, 1), (1278, 1, 1, 1))
    assert_size_stride(primals_625, (469, ), (1, ))
    assert_size_stride(primals_626, (469, ), (1, ))
    assert_size_stride(primals_627, (469, ), (1, ))
    assert_size_stride(primals_628, (469, ), (1, ))
    assert_size_stride(primals_629, (469, ), (1, ))
    assert_size_stride(primals_630, (469, 489, 1, 1), (489, 1, 1, 1))
    assert_size_stride(primals_631, (469, ), (1, ))
    assert_size_stride(primals_632, (469, ), (1, ))
    assert_size_stride(primals_633, (469, ), (1, ))
    assert_size_stride(primals_634, (469, ), (1, ))
    assert_size_stride(primals_635, (469, ), (1, ))
    assert_size_stride(primals_636, (189, 469, 1, 1), (469, 1, 1, 1))
    assert_size_stride(primals_637, (189, ), (1, ))
    assert_size_stride(primals_638, (189, ), (1, ))
    assert_size_stride(primals_639, (189, ), (1, ))
    assert_size_stride(primals_640, (189, ), (1, ))
    assert_size_stride(primals_641, (189, ), (1, ))
    assert_size_stride(primals_642, (105, 189, 1, 1), (189, 1, 1, 1))
    assert_size_stride(primals_643, (105, ), (1, ))
    assert_size_stride(primals_644, (105, ), (1, ))
    assert_size_stride(primals_645, (105, ), (1, ))
    assert_size_stride(primals_646, (105, ), (1, ))
    assert_size_stride(primals_647, (105, ), (1, ))
    assert_size_stride(primals_648, (113, 430, 1, 1), (430, 1, 1, 1))
    assert_size_stride(primals_649, (113, ), (1, ))
    assert_size_stride(primals_650, (113, ), (1, ))
    assert_size_stride(primals_651, (113, ), (1, ))
    assert_size_stride(primals_652, (113, ), (1, ))
    assert_size_stride(primals_653, (113, ), (1, ))
    assert_size_stride(primals_654, (678, 113, 1, 1), (113, 1, 1, 1))
    assert_size_stride(primals_655, (678, ), (1, ))
    assert_size_stride(primals_656, (678, ), (1, ))
    assert_size_stride(primals_657, (678, ), (1, ))
    assert_size_stride(primals_658, (678, ), (1, ))
    assert_size_stride(primals_659, (678, ), (1, ))
    assert_size_stride(primals_660, (678, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_661, (678, ), (1, ))
    assert_size_stride(primals_662, (678, ), (1, ))
    assert_size_stride(primals_663, (678, ), (1, ))
    assert_size_stride(primals_664, (678, ), (1, ))
    assert_size_stride(primals_665, (678, ), (1, ))
    assert_size_stride(primals_666, (325, 678, 1, 1), (678, 1, 1, 1))
    assert_size_stride(primals_667, (325, ), (1, ))
    assert_size_stride(primals_668, (325, ), (1, ))
    assert_size_stride(primals_669, (325, ), (1, ))
    assert_size_stride(primals_670, (325, ), (1, ))
    assert_size_stride(primals_671, (325, ), (1, ))
    assert_size_stride(primals_672, (325, 430, 1, 1), (430, 1, 1, 1))
    assert_size_stride(primals_673, (325, ), (1, ))
    assert_size_stride(primals_674, (325, ), (1, ))
    assert_size_stride(primals_675, (325, ), (1, ))
    assert_size_stride(primals_676, (325, ), (1, ))
    assert_size_stride(primals_677, (325, ), (1, ))
    assert_size_stride(primals_678, (99, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_679, (99, ), (1, ))
    assert_size_stride(primals_680, (99, ), (1, ))
    assert_size_stride(primals_681, (99, ), (1, ))
    assert_size_stride(primals_682, (99, ), (1, ))
    assert_size_stride(primals_683, (99, ), (1, ))
    assert_size_stride(primals_684, (594, 99, 1, 1), (99, 1, 1, 1))
    assert_size_stride(primals_685, (594, ), (1, ))
    assert_size_stride(primals_686, (594, ), (1, ))
    assert_size_stride(primals_687, (594, ), (1, ))
    assert_size_stride(primals_688, (594, ), (1, ))
    assert_size_stride(primals_689, (594, ), (1, ))
    assert_size_stride(primals_690, (594, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_691, (594, ), (1, ))
    assert_size_stride(primals_692, (594, ), (1, ))
    assert_size_stride(primals_693, (594, ), (1, ))
    assert_size_stride(primals_694, (594, ), (1, ))
    assert_size_stride(primals_695, (594, ), (1, ))
    assert_size_stride(primals_696, (207, 594, 1, 1), (594, 1, 1, 1))
    assert_size_stride(primals_697, (207, ), (1, ))
    assert_size_stride(primals_698, (207, ), (1, ))
    assert_size_stride(primals_699, (207, ), (1, ))
    assert_size_stride(primals_700, (207, ), (1, ))
    assert_size_stride(primals_701, (207, ), (1, ))
    assert_size_stride(primals_702, (207, 325, 1, 1), (325, 1, 1, 1))
    assert_size_stride(primals_703, (207, ), (1, ))
    assert_size_stride(primals_704, (207, ), (1, ))
    assert_size_stride(primals_705, (207, ), (1, ))
    assert_size_stride(primals_706, (207, ), (1, ))
    assert_size_stride(primals_707, (207, ), (1, ))
    assert_size_stride(primals_708, (98, 207, 1, 1), (207, 1, 1, 1))
    assert_size_stride(primals_709, (98, ), (1, ))
    assert_size_stride(primals_710, (98, ), (1, ))
    assert_size_stride(primals_711, (98, ), (1, ))
    assert_size_stride(primals_712, (98, ), (1, ))
    assert_size_stride(primals_713, (98, ), (1, ))
    assert_size_stride(primals_714, (47, 98, 1, 1), (98, 1, 1, 1))
    assert_size_stride(primals_715, (47, ), (1, ))
    assert_size_stride(primals_716, (47, ), (1, ))
    assert_size_stride(primals_717, (47, ), (1, ))
    assert_size_stride(primals_718, (47, ), (1, ))
    assert_size_stride(primals_719, (47, ), (1, ))
    assert_size_stride(primals_720, (58, 197, 1, 1), (197, 1, 1, 1))
    assert_size_stride(primals_721, (58, ), (1, ))
    assert_size_stride(primals_722, (58, ), (1, ))
    assert_size_stride(primals_723, (58, ), (1, ))
    assert_size_stride(primals_724, (58, ), (1, ))
    assert_size_stride(primals_725, (58, ), (1, ))
    assert_size_stride(primals_726, (348, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(primals_727, (348, ), (1, ))
    assert_size_stride(primals_728, (348, ), (1, ))
    assert_size_stride(primals_729, (348, ), (1, ))
    assert_size_stride(primals_730, (348, ), (1, ))
    assert_size_stride(primals_731, (348, ), (1, ))
    assert_size_stride(primals_732, (348, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_733, (348, ), (1, ))
    assert_size_stride(primals_734, (348, ), (1, ))
    assert_size_stride(primals_735, (348, ), (1, ))
    assert_size_stride(primals_736, (348, ), (1, ))
    assert_size_stride(primals_737, (348, ), (1, ))
    assert_size_stride(primals_738, (122, 348, 1, 1), (348, 1, 1, 1))
    assert_size_stride(primals_739, (122, ), (1, ))
    assert_size_stride(primals_740, (122, ), (1, ))
    assert_size_stride(primals_741, (122, ), (1, ))
    assert_size_stride(primals_742, (122, ), (1, ))
    assert_size_stride(primals_743, (122, ), (1, ))
    assert_size_stride(primals_744, (122, 197, 1, 1), (197, 1, 1, 1))
    assert_size_stride(primals_745, (122, ), (1, ))
    assert_size_stride(primals_746, (122, ), (1, ))
    assert_size_stride(primals_747, (122, ), (1, ))
    assert_size_stride(primals_748, (122, ), (1, ))
    assert_size_stride(primals_749, (122, ), (1, ))
    assert_size_stride(primals_750, (52, 122, 1, 1), (122, 1, 1, 1))
    assert_size_stride(primals_751, (52, ), (1, ))
    assert_size_stride(primals_752, (52, ), (1, ))
    assert_size_stride(primals_753, (52, ), (1, ))
    assert_size_stride(primals_754, (52, ), (1, ))
    assert_size_stride(primals_755, (52, ), (1, ))
    assert_size_stride(primals_756, (312, 52, 1, 1), (52, 1, 1, 1))
    assert_size_stride(primals_757, (312, ), (1, ))
    assert_size_stride(primals_758, (312, ), (1, ))
    assert_size_stride(primals_759, (312, ), (1, ))
    assert_size_stride(primals_760, (312, ), (1, ))
    assert_size_stride(primals_761, (312, ), (1, ))
    assert_size_stride(primals_762, (312, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_763, (312, ), (1, ))
    assert_size_stride(primals_764, (312, ), (1, ))
    assert_size_stride(primals_765, (312, ), (1, ))
    assert_size_stride(primals_766, (312, ), (1, ))
    assert_size_stride(primals_767, (312, ), (1, ))
    assert_size_stride(primals_768, (87, 312, 1, 1), (312, 1, 1, 1))
    assert_size_stride(primals_769, (87, ), (1, ))
    assert_size_stride(primals_770, (87, ), (1, ))
    assert_size_stride(primals_771, (87, ), (1, ))
    assert_size_stride(primals_772, (87, ), (1, ))
    assert_size_stride(primals_773, (87, ), (1, ))
    assert_size_stride(primals_774, (87, 122, 1, 1), (122, 1, 1, 1))
    assert_size_stride(primals_775, (87, ), (1, ))
    assert_size_stride(primals_776, (87, ), (1, ))
    assert_size_stride(primals_777, (87, ), (1, ))
    assert_size_stride(primals_778, (87, ), (1, ))
    assert_size_stride(primals_779, (87, ), (1, ))
    assert_size_stride(primals_780, (47, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(primals_781, (47, ), (1, ))
    assert_size_stride(primals_782, (47, ), (1, ))
    assert_size_stride(primals_783, (47, ), (1, ))
    assert_size_stride(primals_784, (47, ), (1, ))
    assert_size_stride(primals_785, (47, ), (1, ))
    assert_size_stride(primals_786, (282, 47, 1, 1), (47, 1, 1, 1))
    assert_size_stride(primals_787, (282, ), (1, ))
    assert_size_stride(primals_788, (282, ), (1, ))
    assert_size_stride(primals_789, (282, ), (1, ))
    assert_size_stride(primals_790, (282, ), (1, ))
    assert_size_stride(primals_791, (282, ), (1, ))
    assert_size_stride(primals_792, (282, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_793, (282, ), (1, ))
    assert_size_stride(primals_794, (282, ), (1, ))
    assert_size_stride(primals_795, (282, ), (1, ))
    assert_size_stride(primals_796, (282, ), (1, ))
    assert_size_stride(primals_797, (282, ), (1, ))
    assert_size_stride(primals_798, (93, 282, 1, 1), (282, 1, 1, 1))
    assert_size_stride(primals_799, (93, ), (1, ))
    assert_size_stride(primals_800, (93, ), (1, ))
    assert_size_stride(primals_801, (93, ), (1, ))
    assert_size_stride(primals_802, (93, ), (1, ))
    assert_size_stride(primals_803, (93, ), (1, ))
    assert_size_stride(primals_804, (93, 87, 1, 1), (87, 1, 1, 1))
    assert_size_stride(primals_805, (93, ), (1, ))
    assert_size_stride(primals_806, (93, ), (1, ))
    assert_size_stride(primals_807, (93, ), (1, ))
    assert_size_stride(primals_808, (93, ), (1, ))
    assert_size_stride(primals_809, (93, ), (1, ))
    assert_size_stride(primals_810, (75, 93, 1, 1), (93, 1, 1, 1))
    assert_size_stride(primals_811, (75, ), (1, ))
    assert_size_stride(primals_812, (75, ), (1, ))
    assert_size_stride(primals_813, (75, ), (1, ))
    assert_size_stride(primals_814, (75, ), (1, ))
    assert_size_stride(primals_815, (75, ), (1, ))
    assert_size_stride(primals_816, (588, 98, 1, 1), (98, 1, 1, 1))
    assert_size_stride(primals_817, (588, ), (1, ))
    assert_size_stride(primals_818, (588, ), (1, ))
    assert_size_stride(primals_819, (588, ), (1, ))
    assert_size_stride(primals_820, (588, ), (1, ))
    assert_size_stride(primals_821, (588, ), (1, ))
    assert_size_stride(primals_822, (588, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_823, (588, ), (1, ))
    assert_size_stride(primals_824, (588, ), (1, ))
    assert_size_stride(primals_825, (588, ), (1, ))
    assert_size_stride(primals_826, (588, ), (1, ))
    assert_size_stride(primals_827, (588, ), (1, ))
    assert_size_stride(primals_828, (183, 588, 1, 1), (588, 1, 1, 1))
    assert_size_stride(primals_829, (183, ), (1, ))
    assert_size_stride(primals_830, (183, ), (1, ))
    assert_size_stride(primals_831, (183, ), (1, ))
    assert_size_stride(primals_832, (183, ), (1, ))
    assert_size_stride(primals_833, (183, ), (1, ))
    assert_size_stride(primals_834, (183, 98, 1, 1), (98, 1, 1, 1))
    assert_size_stride(primals_835, (183, ), (1, ))
    assert_size_stride(primals_836, (183, ), (1, ))
    assert_size_stride(primals_837, (183, ), (1, ))
    assert_size_stride(primals_838, (183, ), (1, ))
    assert_size_stride(primals_839, (183, ), (1, ))
    assert_size_stride(primals_840, (75, 183, 1, 1), (183, 1, 1, 1))
    assert_size_stride(primals_841, (75, ), (1, ))
    assert_size_stride(primals_842, (75, ), (1, ))
    assert_size_stride(primals_843, (75, ), (1, ))
    assert_size_stride(primals_844, (75, ), (1, ))
    assert_size_stride(primals_845, (75, ), (1, ))
    assert_size_stride(primals_846, (1134, 189, 1, 1), (189, 1, 1, 1))
    assert_size_stride(primals_847, (1134, ), (1, ))
    assert_size_stride(primals_848, (1134, ), (1, ))
    assert_size_stride(primals_849, (1134, ), (1, ))
    assert_size_stride(primals_850, (1134, ), (1, ))
    assert_size_stride(primals_851, (1134, ), (1, ))
    assert_size_stride(primals_852, (1134, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_853, (1134, ), (1, ))
    assert_size_stride(primals_854, (1134, ), (1, ))
    assert_size_stride(primals_855, (1134, ), (1, ))
    assert_size_stride(primals_856, (1134, ), (1, ))
    assert_size_stride(primals_857, (1134, ), (1, ))
    assert_size_stride(primals_858, (462, 1134, 1, 1), (1134, 1, 1, 1))
    assert_size_stride(primals_859, (462, ), (1, ))
    assert_size_stride(primals_860, (462, ), (1, ))
    assert_size_stride(primals_861, (462, ), (1, ))
    assert_size_stride(primals_862, (462, ), (1, ))
    assert_size_stride(primals_863, (462, ), (1, ))
    assert_size_stride(primals_864, (462, 189, 1, 1), (189, 1, 1, 1))
    assert_size_stride(primals_865, (462, ), (1, ))
    assert_size_stride(primals_866, (462, ), (1, ))
    assert_size_stride(primals_867, (462, ), (1, ))
    assert_size_stride(primals_868, (462, ), (1, ))
    assert_size_stride(primals_869, (462, ), (1, ))
    assert_size_stride(primals_870, (75, 462, 1, 1), (462, 1, 1, 1))
    assert_size_stride(primals_871, (75, ), (1, ))
    assert_size_stride(primals_872, (75, ), (1, ))
    assert_size_stride(primals_873, (75, ), (1, ))
    assert_size_stride(primals_874, (75, ), (1, ))
    assert_size_stride(primals_875, (75, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf316 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_0.run(buf316, 4, grid=grid(4), stream=stream0)
        buf318 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_0.run(buf318, 4, grid=grid(4), stream=stream0)
        buf312 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(buf312, 4, grid=grid(4), stream=stream0)
        buf313 = empty_strided_cuda((4, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_2.run(buf313, 4, grid=grid(4), stream=stream0)
        buf314 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(buf314, 4, grid=grid(4), stream=stream0)
        buf315 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_2.run(buf315, 4, grid=grid(4), stream=stream0)
        buf358 = empty_strided_cuda((8, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_3.run(buf358, 8, grid=grid(8), stream=stream0)
        buf360 = empty_strided_cuda((8, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.sub, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_arange_clamp_mul_sub_3.run(buf360, 8, grid=grid(8), stream=stream0)
        buf354 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf354, 8, grid=grid(8), stream=stream0)
        buf355 = empty_strided_cuda((8, 1), (1, 1), torch.int64)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_5.run(buf355, 8, grid=grid(8), stream=stream0)
        buf356 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.arange, aten._to_copy, aten.mul, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_4.run(buf356, 8, grid=grid(8), stream=stream0)
        buf357 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten.add, aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_5.run(buf357, 8, grid=grid(8), stream=stream0)
        buf0 = empty_strided_cuda((12, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_1, buf0, 36, 9, grid=grid(36, 9), stream=stream0)
        del primals_1
        buf2 = empty_strided_cuda((24, 12, 3, 3), (108, 1, 36, 12), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_8, buf2, 288, 9, grid=grid(288, 9), stream=stream0)
        del primals_8
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_8.run(primals_3, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf1, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 12, 64, 64), (49152, 1, 768, 12))
        buf4 = buf3; del buf3  # reuse
        buf5 = empty_strided_cuda((4, 12, 64, 64), (49152, 1, 768, 12), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9.run(buf4, primals_2, primals_4, primals_5, primals_6, primals_7, buf5, 196608, grid=grid(196608), stream=stream0)
        del primals_2
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, buf2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5, input_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10.run(buf7, primals_9, primals_10, primals_11, primals_12, primals_13, buf8, 98304, grid=grid(98304), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_14, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 7, 32, 32), (7168, 1, 224, 7))
        buf10 = buf9; del buf9  # reuse
        buf11 = empty_strided_cuda((4, 7, 32, 32), (7168, 1, 224, 7), torch.float32)
        # Topologically Sorted Source Nodes: [input_7, input_8, input_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11.run(buf10, primals_15, primals_16, primals_17, primals_18, primals_19, buf11, 28672, grid=grid(28672), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf8, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 24, 32, 32), (24576, 1, 768, 24))
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 42, 32, 32), (43008, 1, 1344, 42))
        buf13 = buf12; del buf12  # reuse
        buf14 = empty_strided_cuda((4, 42, 32, 32), (43008, 1, 1344, 42), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11, input_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12.run(buf13, primals_21, primals_22, primals_23, primals_24, primals_25, buf14, 172032, grid=grid(172032), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=42, bias=None)
        assert_size_stride(buf15, (4, 42, 32, 32), (43008, 1, 1344, 42))
        buf16 = buf15; del buf15  # reuse
        buf17 = empty_strided_cuda((4, 42, 32, 32), (43008, 1, 1344, 42), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14, input_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12.run(buf16, primals_27, primals_28, primals_29, primals_30, primals_31, buf17, 172032, grid=grid(172032), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 24, 32, 32), (24576, 1, 768, 24))
        buf19 = buf18; del buf18  # reuse
        buf21 = buf20; del buf20  # reuse
        buf22 = empty_strided_cuda((4, 24, 32, 32), (24576, 1, 768, 24), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17, input_18, input_19, out], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_13.run(buf19, buf21, primals_33, primals_39, primals_34, primals_35, primals_36, primals_37, primals_40, primals_41, primals_42, primals_43, buf22, 98304, grid=grid(98304), stream=stream0)
        del primals_33
        del primals_37
        del primals_39
        del primals_43
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 144, 32, 32), (147456, 1, 4608, 144))
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided_cuda((4, 144, 32, 32), (147456, 1, 4608, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_20, input_21, input_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14.run(buf24, primals_45, primals_46, primals_47, primals_48, primals_49, buf25, 589824, grid=grid(589824), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_50, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf26, (4, 144, 16, 16), (36864, 1, 2304, 144))
        buf27 = buf26; del buf26  # reuse
        buf28 = empty_strided_cuda((4, 144, 16, 16), (36864, 1, 2304, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24, input_25], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15.run(buf27, primals_51, primals_52, primals_53, primals_54, primals_55, buf28, 147456, grid=grid(147456), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_56, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 70, 16, 16), (17920, 1, 1120, 70))
        buf30 = buf29; del buf29  # reuse
        buf31 = empty_strided_cuda((4, 70, 16, 16), (17920, 1, 1120, 70), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_16.run(buf30, primals_57, primals_58, primals_59, primals_60, primals_61, buf31, 71680, grid=grid(71680), stream=stream0)
        del primals_57
        del primals_61
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 25, 16, 16), (6400, 1, 400, 25))
        buf33 = buf32; del buf32  # reuse
        buf34 = empty_strided_cuda((4, 25, 16, 16), (6400, 1, 400, 25), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29, input_30], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_17.run(buf33, primals_63, primals_64, primals_65, primals_66, primals_67, buf34, 25600, grid=grid(25600), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf31, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 70, 16, 16), (17920, 1, 1120, 70))
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 150, 16, 16), (38400, 1, 2400, 150))
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided_cuda((4, 150, 16, 16), (38400, 1, 2400, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_31, input_32, input_33], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18.run(buf36, primals_69, primals_70, primals_71, primals_72, primals_73, buf37, 153600, grid=grid(153600), stream=stream0)
        del primals_69
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_74, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=150, bias=None)
        assert_size_stride(buf38, (4, 150, 16, 16), (38400, 1, 2400, 150))
        buf39 = buf38; del buf38  # reuse
        buf40 = empty_strided_cuda((4, 150, 16, 16), (38400, 1, 2400, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35, input_36], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18.run(buf39, primals_75, primals_76, primals_77, primals_78, primals_79, buf40, 153600, grid=grid(153600), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [input_37], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 70, 16, 16), (17920, 1, 1120, 70))
        buf42 = buf41; del buf41  # reuse
        buf44 = buf43; del buf43  # reuse
        buf45 = empty_strided_cuda((4, 70, 16, 16), (17920, 1, 1120, 70), torch.float32)
        # Topologically Sorted Source Nodes: [input_37, input_38, input_39, input_40, out_1], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19.run(buf42, buf44, primals_81, primals_87, primals_82, primals_83, primals_84, primals_85, primals_88, primals_89, primals_90, primals_91, buf45, 71680, grid=grid(71680), stream=stream0)
        del primals_81
        del primals_85
        del primals_87
        del primals_91
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 24, 16, 16), (6144, 1, 384, 24))
        buf47 = buf46; del buf46  # reuse
        buf48 = empty_strided_cuda((4, 24, 16, 16), (6144, 1, 384, 24), torch.float32)
        # Topologically Sorted Source Nodes: [input_41, input_42, input_43], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20.run(buf47, primals_93, primals_94, primals_95, primals_96, primals_97, buf48, 24576, grid=grid(24576), stream=stream0)
        del primals_93
        # Topologically Sorted Source Nodes: [input_52], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf45, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 70, 16, 16), (17920, 1, 1120, 70))
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_98, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 144, 16, 16), (36864, 1, 2304, 144))
        buf50 = buf49; del buf49  # reuse
        buf51 = empty_strided_cuda((4, 144, 16, 16), (36864, 1, 2304, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, input_45, input_46], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15.run(buf50, primals_99, primals_100, primals_101, primals_102, primals_103, buf51, 147456, grid=grid(147456), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=144, bias=None)
        assert_size_stride(buf52, (4, 144, 16, 16), (36864, 1, 2304, 144))
        buf53 = buf52; del buf52  # reuse
        buf54 = empty_strided_cuda((4, 144, 16, 16), (36864, 1, 2304, 144), torch.float32)
        # Topologically Sorted Source Nodes: [input_47, input_48, input_49], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15.run(buf53, primals_105, primals_106, primals_107, primals_108, primals_109, buf54, 147456, grid=grid(147456), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_110, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 70, 16, 16), (17920, 1, 1120, 70))
        buf56 = buf55; del buf55  # reuse
        buf58 = buf57; del buf57  # reuse
        buf59 = empty_strided_cuda((4, 70, 16, 16), (17920, 1, 1120, 70), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51, input_52, input_53, out_2], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_19.run(buf56, buf58, primals_111, primals_117, primals_112, primals_113, primals_114, primals_115, primals_118, primals_119, primals_120, primals_121, buf59, 71680, grid=grid(71680), stream=stream0)
        del primals_111
        del primals_115
        del primals_117
        del primals_121
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 420, 16, 16), (107520, 1, 6720, 420))
        buf61 = buf60; del buf60  # reuse
        buf62 = empty_strided_cuda((4, 420, 16, 16), (107520, 1, 6720, 420), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, input_55, input_56], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_21.run(buf61, primals_123, primals_124, primals_125, primals_126, primals_127, buf62, 430080, grid=grid(430080), stream=stream0)
        del primals_123
        # Topologically Sorted Source Nodes: [input_57], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_128, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=420, bias=None)
        assert_size_stride(buf63, (4, 420, 8, 8), (26880, 1, 3360, 420))
        buf64 = buf63; del buf63  # reuse
        buf65 = empty_strided_cuda((4, 420, 8, 8), (26880, 1, 3360, 420), torch.float32)
        # Topologically Sorted Source Nodes: [input_57, input_58, input_59], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_22.run(buf64, primals_129, primals_130, primals_131, primals_132, primals_133, buf65, 107520, grid=grid(107520), stream=stream0)
        del primals_129
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_134, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 150, 8, 8), (9600, 1, 1200, 150))
        buf67 = buf66; del buf66  # reuse
        buf68 = empty_strided_cuda((4, 150, 8, 8), (9600, 1, 1200, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_23.run(buf67, primals_135, primals_136, primals_137, primals_138, primals_139, buf68, 38400, grid=grid(38400), stream=stream0)
        del primals_135
        del primals_139
        # Topologically Sorted Source Nodes: [input_62], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_140, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 56, 8, 8), (3584, 1, 448, 56))
        buf70 = buf69; del buf69  # reuse
        buf71 = empty_strided_cuda((4, 56, 8, 8), (3584, 1, 448, 56), torch.float32)
        # Topologically Sorted Source Nodes: [input_62, input_63, input_64], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24.run(buf70, primals_141, primals_142, primals_143, primals_144, primals_145, buf71, 14336, grid=grid(14336), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf68, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 150, 8, 8), (9600, 1, 1200, 150))
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_146, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 336, 8, 8), (21504, 1, 2688, 336))
        buf73 = buf72; del buf72  # reuse
        buf74 = empty_strided_cuda((4, 336, 8, 8), (21504, 1, 2688, 336), torch.float32)
        # Topologically Sorted Source Nodes: [input_65, input_66, input_67], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25.run(buf73, primals_147, primals_148, primals_149, primals_150, primals_151, buf74, 86016, grid=grid(86016), stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_152, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=336, bias=None)
        assert_size_stride(buf75, (4, 336, 8, 8), (21504, 1, 2688, 336))
        buf76 = buf75; del buf75  # reuse
        buf77 = empty_strided_cuda((4, 336, 8, 8), (21504, 1, 2688, 336), torch.float32)
        # Topologically Sorted Source Nodes: [input_68, input_69, input_70], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_25.run(buf76, primals_153, primals_154, primals_155, primals_156, primals_157, buf77, 86016, grid=grid(86016), stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_158, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 150, 8, 8), (9600, 1, 1200, 150))
        buf79 = buf78; del buf78  # reuse
        buf81 = buf80; del buf80  # reuse
        buf82 = empty_strided_cuda((4, 150, 8, 8), (9600, 1, 1200, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, input_72, input_73, input_74, out_3], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26.run(buf79, buf81, primals_159, primals_165, primals_160, primals_161, primals_162, primals_163, primals_166, primals_167, primals_168, primals_169, buf82, 38400, grid=grid(38400), stream=stream0)
        del primals_159
        del primals_163
        del primals_165
        del primals_169
        # Topologically Sorted Source Nodes: [input_75], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 150, 8, 8), (9600, 1, 1200, 150))
        buf84 = buf83; del buf83  # reuse
        buf85 = empty_strided_cuda((4, 150, 8, 8), (9600, 1, 1200, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_75, input_76, input_77], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_27.run(buf84, primals_171, primals_172, primals_173, primals_174, primals_175, buf85, 38400, grid=grid(38400), stream=stream0)
        del primals_171
        buf86 = empty_strided_cuda((4, 150, 1, 1), (150, 1, 600, 600), torch.float32)
        buf87 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [adaptive_avg_pool2d], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_mean_28.run(buf87, buf85, 600, 64, grid=grid(600), stream=stream0)
        buf88 = empty_strided_cuda((4, 18), (18, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf87, (4, 150), (150, 1), 0), reinterpret_tensor(primals_176, (150, 18), (1, 150), 0), out=buf88)
        buf89 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [input_78, input_79], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_29.run(buf89, primals_177, 72, grid=grid(72), stream=stream0)
        del primals_177
        buf90 = empty_strided_cuda((4, 150), (150, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_80], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_179, buf89, reinterpret_tensor(primals_178, (18, 150), (1, 18), 0), alpha=1, beta=1, out=buf90)
        del primals_179
        buf91 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_30.run(buf91, buf90, 38400, grid=grid(38400), stream=stream0)
        # Topologically Sorted Source Nodes: [input_83], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 73, 8, 8), (4672, 1, 584, 73))
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided_cuda((4, 73, 8, 8), (4672, 1, 584, 73), torch.float32)
        # Topologically Sorted Source Nodes: [input_83, input_84, input_85], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31.run(buf93, primals_181, primals_182, primals_183, primals_184, primals_185, buf94, 18688, grid=grid(18688), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf91, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 150, 8, 8), (9600, 1, 1200, 150))
        # Topologically Sorted Source Nodes: [input_86], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 438, 8, 8), (28032, 1, 3504, 438))
        buf96 = buf95; del buf95  # reuse
        buf97 = empty_strided_cuda((4, 438, 8, 8), (28032, 1, 3504, 438), torch.float32)
        # Topologically Sorted Source Nodes: [input_86, input_87, input_88], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32.run(buf96, primals_187, primals_188, primals_189, primals_190, primals_191, buf97, 112128, grid=grid(112128), stream=stream0)
        del primals_187
        # Topologically Sorted Source Nodes: [input_89], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_192, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=438, bias=None)
        assert_size_stride(buf98, (4, 438, 8, 8), (28032, 1, 3504, 438))
        buf99 = buf98; del buf98  # reuse
        buf100 = empty_strided_cuda((4, 438, 8, 8), (28032, 1, 3504, 438), torch.float32)
        # Topologically Sorted Source Nodes: [input_89, input_90, input_91], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32.run(buf99, primals_193, primals_194, primals_195, primals_196, primals_197, buf100, 112128, grid=grid(112128), stream=stream0)
        del primals_193
        # Topologically Sorted Source Nodes: [input_92], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 150, 8, 8), (9600, 1, 1200, 150))
        buf102 = buf101; del buf101  # reuse
        buf104 = buf103; del buf103  # reuse
        buf105 = empty_strided_cuda((4, 150, 8, 8), (9600, 1, 1200, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, input_93, input_94, input_95, out_4], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26.run(buf102, buf104, primals_199, primals_205, primals_200, primals_201, primals_202, primals_203, primals_206, primals_207, primals_208, primals_209, buf105, 38400, grid=grid(38400), stream=stream0)
        del primals_199
        del primals_203
        del primals_205
        del primals_209
        # Topologically Sorted Source Nodes: [input_96], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 71, 8, 8), (4544, 1, 568, 71))
        buf107 = buf106; del buf106  # reuse
        buf108 = empty_strided_cuda((4, 71, 8, 8), (4544, 1, 568, 71), torch.float32)
        # Topologically Sorted Source Nodes: [input_96, input_97, input_98], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33.run(buf107, primals_211, primals_212, primals_213, primals_214, primals_215, buf108, 18176, grid=grid(18176), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [input_107], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf105, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 150, 8, 8), (9600, 1, 1200, 150))
        # Topologically Sorted Source Nodes: [input_99], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 426, 8, 8), (27264, 1, 3408, 426))
        buf110 = buf109; del buf109  # reuse
        buf111 = empty_strided_cuda((4, 426, 8, 8), (27264, 1, 3408, 426), torch.float32)
        # Topologically Sorted Source Nodes: [input_99, input_100, input_101], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34.run(buf110, primals_217, primals_218, primals_219, primals_220, primals_221, buf111, 109056, grid=grid(109056), stream=stream0)
        del primals_217
        # Topologically Sorted Source Nodes: [input_102], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf111, primals_222, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=426, bias=None)
        assert_size_stride(buf112, (4, 426, 8, 8), (27264, 1, 3408, 426))
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided_cuda((4, 426, 8, 8), (27264, 1, 3408, 426), torch.float32)
        # Topologically Sorted Source Nodes: [input_102, input_103, input_104], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_34.run(buf113, primals_223, primals_224, primals_225, primals_226, primals_227, buf114, 109056, grid=grid(109056), stream=stream0)
        del primals_223
        # Topologically Sorted Source Nodes: [input_105], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 150, 8, 8), (9600, 1, 1200, 150))
        buf116 = buf115; del buf115  # reuse
        buf118 = buf117; del buf117  # reuse
        buf119 = empty_strided_cuda((4, 150, 8, 8), (9600, 1, 1200, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_105, input_106, input_107, input_108, out_5], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26.run(buf116, buf118, primals_229, primals_235, primals_230, primals_231, primals_232, primals_233, primals_236, primals_237, primals_238, primals_239, buf119, 38400, grid=grid(38400), stream=stream0)
        del primals_229
        del primals_233
        del primals_235
        del primals_239
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 75, 8, 8), (4800, 1, 600, 75))
        buf121 = buf120; del buf120  # reuse
        buf122 = empty_strided_cuda((4, 75, 8, 8), (4800, 1, 600, 75), torch.float32)
        # Topologically Sorted Source Nodes: [input_109, input_110, input_111], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35.run(buf121, primals_241, primals_242, primals_243, primals_244, primals_245, buf122, 19200, grid=grid(19200), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [input_120], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf119, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 150, 8, 8), (9600, 1, 1200, 150))
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 450, 8, 8), (28800, 1, 3600, 450))
        buf124 = buf123; del buf123  # reuse
        buf125 = empty_strided_cuda((4, 450, 8, 8), (28800, 1, 3600, 450), torch.float32)
        # Topologically Sorted Source Nodes: [input_112, input_113, input_114], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36.run(buf124, primals_247, primals_248, primals_249, primals_250, primals_251, buf125, 115200, grid=grid(115200), stream=stream0)
        del primals_247
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_252, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=450, bias=None)
        assert_size_stride(buf126, (4, 450, 8, 8), (28800, 1, 3600, 450))
        buf127 = buf126; del buf126  # reuse
        buf128 = empty_strided_cuda((4, 450, 8, 8), (28800, 1, 3600, 450), torch.float32)
        # Topologically Sorted Source Nodes: [input_115, input_116, input_117], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_36.run(buf127, primals_253, primals_254, primals_255, primals_256, primals_257, buf128, 115200, grid=grid(115200), stream=stream0)
        del primals_253
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (4, 150, 8, 8), (9600, 1, 1200, 150))
        buf130 = buf129; del buf129  # reuse
        buf132 = buf131; del buf131  # reuse
        buf133 = empty_strided_cuda((4, 150, 8, 8), (9600, 1, 1200, 150), torch.float32)
        # Topologically Sorted Source Nodes: [input_118, input_119, input_120, input_121, out_6], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_26.run(buf130, buf132, primals_259, primals_265, primals_260, primals_261, primals_262, primals_263, primals_266, primals_267, primals_268, primals_269, buf133, 38400, grid=grid(38400), stream=stream0)
        del primals_259
        del primals_263
        del primals_265
        del primals_269
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_270, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 900, 8, 8), (57600, 1, 7200, 900))
        buf135 = buf134; del buf134  # reuse
        buf136 = empty_strided_cuda((4, 900, 8, 8), (57600, 1, 7200, 900), torch.float32)
        # Topologically Sorted Source Nodes: [input_122, input_123, input_124], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37.run(buf135, primals_271, primals_272, primals_273, primals_274, primals_275, buf136, 230400, grid=grid(230400), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [input_125], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_276, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=900, bias=None)
        assert_size_stride(buf137, (4, 900, 4, 4), (14400, 1, 3600, 900))
        buf138 = buf137; del buf137  # reuse
        buf139 = empty_strided_cuda((4, 900, 4, 4), (14400, 1, 3600, 900), torch.float32)
        # Topologically Sorted Source Nodes: [input_125, input_126, input_127], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_38.run(buf138, primals_277, primals_278, primals_279, primals_280, primals_281, buf139, 57600, grid=grid(57600), stream=stream0)
        del primals_277
        # Topologically Sorted Source Nodes: [input_128], Original ATen: [aten.convolution]
        buf140 = extern_kernels.convolution(buf139, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf141 = buf140; del buf140  # reuse
        buf142 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_128, input_129], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_39.run(buf141, primals_283, primals_284, primals_285, primals_286, primals_287, buf142, 20800, grid=grid(20800), stream=stream0)
        del primals_283
        del primals_287
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_288, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 132, 4, 4), (2112, 1, 528, 132))
        buf144 = buf143; del buf143  # reuse
        buf145 = empty_strided_cuda((4, 132, 4, 4), (2112, 1, 528, 132), torch.float32)
        # Topologically Sorted Source Nodes: [input_130, input_131, input_132], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_40.run(buf144, primals_289, primals_290, primals_291, primals_292, primals_293, buf145, 8448, grid=grid(8448), stream=stream0)
        del primals_289
        # Topologically Sorted Source Nodes: [input_141], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf142, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_133], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_294, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 792, 4, 4), (12672, 1, 3168, 792))
        buf147 = buf146; del buf146  # reuse
        buf148 = empty_strided_cuda((4, 792, 4, 4), (12672, 1, 3168, 792), torch.float32)
        # Topologically Sorted Source Nodes: [input_133, input_134, input_135], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_41.run(buf147, primals_295, primals_296, primals_297, primals_298, primals_299, buf148, 50688, grid=grid(50688), stream=stream0)
        del primals_295
        # Topologically Sorted Source Nodes: [input_136], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_300, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=792, bias=None)
        assert_size_stride(buf149, (4, 792, 4, 4), (12672, 1, 3168, 792))
        buf150 = buf149; del buf149  # reuse
        buf151 = empty_strided_cuda((4, 792, 4, 4), (12672, 1, 3168, 792), torch.float32)
        # Topologically Sorted Source Nodes: [input_136, input_137, input_138], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_41.run(buf150, primals_301, primals_302, primals_303, primals_304, primals_305, buf151, 50688, grid=grid(50688), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [input_139], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_306, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf153 = buf152; del buf152  # reuse
        buf155 = buf154; del buf154  # reuse
        buf156 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_139, input_140, input_141, input_142, out_7], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf153, buf155, primals_307, primals_313, primals_308, primals_309, primals_310, primals_311, primals_314, primals_315, primals_316, primals_317, buf156, 20800, grid=grid(20800), stream=stream0)
        del primals_307
        del primals_311
        del primals_313
        del primals_317
        # Topologically Sorted Source Nodes: [input_143], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (4, 124, 4, 4), (1984, 1, 496, 124))
        buf158 = buf157; del buf157  # reuse
        buf159 = empty_strided_cuda((4, 124, 4, 4), (1984, 1, 496, 124), torch.float32)
        # Topologically Sorted Source Nodes: [input_143, input_144, input_145], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43.run(buf158, primals_319, primals_320, primals_321, primals_322, primals_323, buf159, 7936, grid=grid(7936), stream=stream0)
        del primals_319
        # Topologically Sorted Source Nodes: [input_154], Original ATen: [aten.convolution]
        buf168 = extern_kernels.convolution(buf156, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf168, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_146], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_324, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (4, 744, 4, 4), (11904, 1, 2976, 744))
        buf161 = buf160; del buf160  # reuse
        buf162 = empty_strided_cuda((4, 744, 4, 4), (11904, 1, 2976, 744), torch.float32)
        # Topologically Sorted Source Nodes: [input_146, input_147, input_148], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44.run(buf161, primals_325, primals_326, primals_327, primals_328, primals_329, buf162, 47616, grid=grid(47616), stream=stream0)
        del primals_325
        # Topologically Sorted Source Nodes: [input_149], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_330, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=744, bias=None)
        assert_size_stride(buf163, (4, 744, 4, 4), (11904, 1, 2976, 744))
        buf164 = buf163; del buf163  # reuse
        buf165 = empty_strided_cuda((4, 744, 4, 4), (11904, 1, 2976, 744), torch.float32)
        # Topologically Sorted Source Nodes: [input_149, input_150, input_151], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_44.run(buf164, primals_331, primals_332, primals_333, primals_334, primals_335, buf165, 47616, grid=grid(47616), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [input_152], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf167 = buf166; del buf166  # reuse
        buf169 = buf168; del buf168  # reuse
        buf170 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_152, input_153, input_154, input_155, out_8], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf167, buf169, primals_337, primals_343, primals_338, primals_339, primals_340, primals_341, primals_344, primals_345, primals_346, primals_347, buf170, 20800, grid=grid(20800), stream=stream0)
        del primals_337
        del primals_341
        del primals_343
        del primals_347
        # Topologically Sorted Source Nodes: [input_156], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_348, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 141, 4, 4), (2256, 1, 564, 141))
        buf172 = buf171; del buf171  # reuse
        buf173 = empty_strided_cuda((4, 141, 4, 4), (2256, 1, 564, 141), torch.float32)
        # Topologically Sorted Source Nodes: [input_156, input_157, input_158], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_45.run(buf172, primals_349, primals_350, primals_351, primals_352, primals_353, buf173, 9024, grid=grid(9024), stream=stream0)
        del primals_349
        # Topologically Sorted Source Nodes: [input_167], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf170, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_159], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, primals_354, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (4, 846, 4, 4), (13536, 1, 3384, 846))
        buf175 = buf174; del buf174  # reuse
        buf176 = empty_strided_cuda((4, 846, 4, 4), (13536, 1, 3384, 846), torch.float32)
        # Topologically Sorted Source Nodes: [input_159, input_160, input_161], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_46.run(buf175, primals_355, primals_356, primals_357, primals_358, primals_359, buf176, 54144, grid=grid(54144), stream=stream0)
        del primals_355
        # Topologically Sorted Source Nodes: [input_162], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=846, bias=None)
        assert_size_stride(buf177, (4, 846, 4, 4), (13536, 1, 3384, 846))
        buf178 = buf177; del buf177  # reuse
        buf179 = empty_strided_cuda((4, 846, 4, 4), (13536, 1, 3384, 846), torch.float32)
        # Topologically Sorted Source Nodes: [input_162, input_163, input_164], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_46.run(buf178, primals_361, primals_362, primals_363, primals_364, primals_365, buf179, 54144, grid=grid(54144), stream=stream0)
        del primals_361
        # Topologically Sorted Source Nodes: [input_165], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_366, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf181 = buf180; del buf180  # reuse
        buf183 = buf182; del buf182  # reuse
        buf184 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_165, input_166, input_167, input_168, out_9], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf181, buf183, primals_367, primals_373, primals_368, primals_369, primals_370, primals_371, primals_374, primals_375, primals_376, primals_377, buf184, 20800, grid=grid(20800), stream=stream0)
        del primals_367
        del primals_371
        del primals_373
        del primals_377
        # Topologically Sorted Source Nodes: [input_169], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_378, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (4, 140, 4, 4), (2240, 1, 560, 140))
        buf186 = buf185; del buf185  # reuse
        buf187 = empty_strided_cuda((4, 140, 4, 4), (2240, 1, 560, 140), torch.float32)
        # Topologically Sorted Source Nodes: [input_169, input_170, input_171], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_47.run(buf186, primals_379, primals_380, primals_381, primals_382, primals_383, buf187, 8960, grid=grid(8960), stream=stream0)
        del primals_379
        # Topologically Sorted Source Nodes: [input_180], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf184, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_172], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_384, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 840, 4, 4), (13440, 1, 3360, 840))
        buf189 = buf188; del buf188  # reuse
        buf190 = empty_strided_cuda((4, 840, 4, 4), (13440, 1, 3360, 840), torch.float32)
        # Topologically Sorted Source Nodes: [input_172, input_173, input_174], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48.run(buf189, primals_385, primals_386, primals_387, primals_388, primals_389, buf190, 53760, grid=grid(53760), stream=stream0)
        del primals_385
        # Topologically Sorted Source Nodes: [input_175], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_390, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=840, bias=None)
        assert_size_stride(buf191, (4, 840, 4, 4), (13440, 1, 3360, 840))
        buf192 = buf191; del buf191  # reuse
        buf193 = empty_strided_cuda((4, 840, 4, 4), (13440, 1, 3360, 840), torch.float32)
        # Topologically Sorted Source Nodes: [input_175, input_176, input_177], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48.run(buf192, primals_391, primals_392, primals_393, primals_394, primals_395, buf193, 53760, grid=grid(53760), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [input_178], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_396, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf195 = buf194; del buf194  # reuse
        buf197 = buf196; del buf196  # reuse
        buf198 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_178, input_179, input_180, input_181, out_10], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf195, buf197, primals_397, primals_403, primals_398, primals_399, primals_400, primals_401, primals_404, primals_405, primals_406, primals_407, buf198, 20800, grid=grid(20800), stream=stream0)
        del primals_397
        del primals_401
        del primals_403
        del primals_407
        # Topologically Sorted Source Nodes: [input_182], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_408, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 137, 4, 4), (2192, 1, 548, 137))
        buf200 = buf199; del buf199  # reuse
        buf201 = empty_strided_cuda((4, 137, 4, 4), (2192, 1, 548, 137), torch.float32)
        # Topologically Sorted Source Nodes: [input_182, input_183, input_184], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_49.run(buf200, primals_409, primals_410, primals_411, primals_412, primals_413, buf201, 8768, grid=grid(8768), stream=stream0)
        del primals_409
        # Topologically Sorted Source Nodes: [input_193], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf198, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_185], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_414, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (4, 822, 4, 4), (13152, 1, 3288, 822))
        buf203 = buf202; del buf202  # reuse
        buf204 = empty_strided_cuda((4, 822, 4, 4), (13152, 1, 3288, 822), torch.float32)
        # Topologically Sorted Source Nodes: [input_185, input_186, input_187], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_50.run(buf203, primals_415, primals_416, primals_417, primals_418, primals_419, buf204, 52608, grid=grid(52608), stream=stream0)
        del primals_415
        # Topologically Sorted Source Nodes: [input_188], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_420, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=822, bias=None)
        assert_size_stride(buf205, (4, 822, 4, 4), (13152, 1, 3288, 822))
        buf206 = buf205; del buf205  # reuse
        buf207 = empty_strided_cuda((4, 822, 4, 4), (13152, 1, 3288, 822), torch.float32)
        # Topologically Sorted Source Nodes: [input_188, input_189, input_190], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_50.run(buf206, primals_421, primals_422, primals_423, primals_424, primals_425, buf207, 52608, grid=grid(52608), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [input_191], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_426, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf209 = buf208; del buf208  # reuse
        buf211 = buf210; del buf210  # reuse
        buf212 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_191, input_192, input_193, input_194, out_11], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf209, buf211, primals_427, primals_433, primals_428, primals_429, primals_430, primals_431, primals_434, primals_435, primals_436, primals_437, buf212, 20800, grid=grid(20800), stream=stream0)
        del primals_427
        del primals_431
        del primals_433
        del primals_437
        # Topologically Sorted Source Nodes: [input_195], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_438, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 135, 4, 4), (2160, 1, 540, 135))
        buf214 = buf213; del buf213  # reuse
        buf215 = empty_strided_cuda((4, 135, 4, 4), (2160, 1, 540, 135), torch.float32)
        # Topologically Sorted Source Nodes: [input_195, input_196, input_197], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_51.run(buf214, primals_439, primals_440, primals_441, primals_442, primals_443, buf215, 8640, grid=grid(8640), stream=stream0)
        del primals_439
        # Topologically Sorted Source Nodes: [input_206], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf212, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_198], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_444, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 810, 4, 4), (12960, 1, 3240, 810))
        buf217 = buf216; del buf216  # reuse
        buf218 = empty_strided_cuda((4, 810, 4, 4), (12960, 1, 3240, 810), torch.float32)
        # Topologically Sorted Source Nodes: [input_198, input_199, input_200], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_52.run(buf217, primals_445, primals_446, primals_447, primals_448, primals_449, buf218, 51840, grid=grid(51840), stream=stream0)
        del primals_445
        # Topologically Sorted Source Nodes: [input_201], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_450, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=810, bias=None)
        assert_size_stride(buf219, (4, 810, 4, 4), (12960, 1, 3240, 810))
        buf220 = buf219; del buf219  # reuse
        buf221 = empty_strided_cuda((4, 810, 4, 4), (12960, 1, 3240, 810), torch.float32)
        # Topologically Sorted Source Nodes: [input_201, input_202, input_203], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_52.run(buf220, primals_451, primals_452, primals_453, primals_454, primals_455, buf221, 51840, grid=grid(51840), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [input_204], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_456, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf223 = buf222; del buf222  # reuse
        buf225 = buf224; del buf224  # reuse
        buf226 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_204, input_205, input_206, input_207, out_12], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf223, buf225, primals_457, primals_463, primals_458, primals_459, primals_460, primals_461, primals_464, primals_465, primals_466, primals_467, buf226, 20800, grid=grid(20800), stream=stream0)
        del primals_457
        del primals_461
        del primals_463
        del primals_467
        # Topologically Sorted Source Nodes: [input_208], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_468, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 133, 4, 4), (2128, 1, 532, 133))
        buf228 = buf227; del buf227  # reuse
        buf229 = empty_strided_cuda((4, 133, 4, 4), (2128, 1, 532, 133), torch.float32)
        # Topologically Sorted Source Nodes: [input_208, input_209, input_210], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_53.run(buf228, primals_469, primals_470, primals_471, primals_472, primals_473, buf229, 8512, grid=grid(8512), stream=stream0)
        del primals_469
        # Topologically Sorted Source Nodes: [input_219], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf226, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_211], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_474, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (4, 798, 4, 4), (12768, 1, 3192, 798))
        buf231 = buf230; del buf230  # reuse
        buf232 = empty_strided_cuda((4, 798, 4, 4), (12768, 1, 3192, 798), torch.float32)
        # Topologically Sorted Source Nodes: [input_211, input_212, input_213], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_54.run(buf231, primals_475, primals_476, primals_477, primals_478, primals_479, buf232, 51072, grid=grid(51072), stream=stream0)
        del primals_475
        # Topologically Sorted Source Nodes: [input_214], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_480, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=798, bias=None)
        assert_size_stride(buf233, (4, 798, 4, 4), (12768, 1, 3192, 798))
        buf234 = buf233; del buf233  # reuse
        buf235 = empty_strided_cuda((4, 798, 4, 4), (12768, 1, 3192, 798), torch.float32)
        # Topologically Sorted Source Nodes: [input_214, input_215, input_216], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_54.run(buf234, primals_481, primals_482, primals_483, primals_484, primals_485, buf235, 51072, grid=grid(51072), stream=stream0)
        del primals_481
        # Topologically Sorted Source Nodes: [input_217], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, primals_486, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf237 = buf236; del buf236  # reuse
        buf239 = buf238; del buf238  # reuse
        buf240 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_217, input_218, input_219, input_220, out_13], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf237, buf239, primals_487, primals_493, primals_488, primals_489, primals_490, primals_491, primals_494, primals_495, primals_496, primals_497, buf240, 20800, grid=grid(20800), stream=stream0)
        del primals_487
        del primals_491
        del primals_493
        del primals_497
        # Topologically Sorted Source Nodes: [input_221], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_498, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 140, 4, 4), (2240, 1, 560, 140))
        buf242 = buf241; del buf241  # reuse
        buf243 = empty_strided_cuda((4, 140, 4, 4), (2240, 1, 560, 140), torch.float32)
        # Topologically Sorted Source Nodes: [input_221, input_222, input_223], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_47.run(buf242, primals_499, primals_500, primals_501, primals_502, primals_503, buf243, 8960, grid=grid(8960), stream=stream0)
        del primals_499
        # Topologically Sorted Source Nodes: [input_232], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf240, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_224], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_504, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (4, 840, 4, 4), (13440, 1, 3360, 840))
        buf245 = buf244; del buf244  # reuse
        buf246 = empty_strided_cuda((4, 840, 4, 4), (13440, 1, 3360, 840), torch.float32)
        # Topologically Sorted Source Nodes: [input_224, input_225, input_226], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48.run(buf245, primals_505, primals_506, primals_507, primals_508, primals_509, buf246, 53760, grid=grid(53760), stream=stream0)
        del primals_505
        # Topologically Sorted Source Nodes: [input_227], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_510, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=840, bias=None)
        assert_size_stride(buf247, (4, 840, 4, 4), (13440, 1, 3360, 840))
        buf248 = buf247; del buf247  # reuse
        buf249 = empty_strided_cuda((4, 840, 4, 4), (13440, 1, 3360, 840), torch.float32)
        # Topologically Sorted Source Nodes: [input_227, input_228, input_229], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_48.run(buf248, primals_511, primals_512, primals_513, primals_514, primals_515, buf249, 53760, grid=grid(53760), stream=stream0)
        del primals_511
        # Topologically Sorted Source Nodes: [input_230], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_516, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf250, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf251 = buf250; del buf250  # reuse
        buf253 = buf252; del buf252  # reuse
        buf254 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_230, input_231, input_232, input_233, out_14], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf251, buf253, primals_517, primals_523, primals_518, primals_519, primals_520, primals_521, primals_524, primals_525, primals_526, primals_527, buf254, 20800, grid=grid(20800), stream=stream0)
        del primals_517
        del primals_521
        del primals_523
        del primals_527
        # Topologically Sorted Source Nodes: [input_234], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_528, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 1950, 4, 4), (31200, 1, 7800, 1950))
        buf256 = buf255; del buf255  # reuse
        buf257 = empty_strided_cuda((4, 1950, 4, 4), (31200, 1, 7800, 1950), torch.float32)
        # Topologically Sorted Source Nodes: [input_234, input_235, input_236], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_55.run(buf256, primals_529, primals_530, primals_531, primals_532, primals_533, buf257, 124800, grid=grid(124800), stream=stream0)
        del primals_529
        # Topologically Sorted Source Nodes: [input_237], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_534, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1950, bias=None)
        assert_size_stride(buf258, (4, 1950, 2, 2), (7800, 1, 3900, 1950))
        buf259 = buf258; del buf258  # reuse
        buf260 = empty_strided_cuda((4, 1950, 2, 2), (7800, 1, 3900, 1950), torch.float32)
        # Topologically Sorted Source Nodes: [input_237, input_238, input_239], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_56.run(buf259, primals_535, primals_536, primals_537, primals_538, primals_539, buf260, 31200, grid=grid(31200), stream=stream0)
        del primals_535
        # Topologically Sorted Source Nodes: [input_240], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_540, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 545, 2, 2), (2180, 1, 1090, 545))
        buf262 = buf261; del buf261  # reuse
        buf263 = empty_strided_cuda((4, 545, 2, 2), (2180, 1, 1090, 545), torch.float32)
        # Topologically Sorted Source Nodes: [input_240, input_241], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_57.run(buf262, primals_541, primals_542, primals_543, primals_544, primals_545, buf263, 8720, grid=grid(8720), stream=stream0)
        del primals_541
        del primals_545
        # Topologically Sorted Source Nodes: [input_242], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_546, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 276, 2, 2), (1104, 1, 552, 276))
        buf265 = buf264; del buf264  # reuse
        buf266 = empty_strided_cuda((4, 276, 2, 2), (1104, 1, 552, 276), torch.float32)
        # Topologically Sorted Source Nodes: [input_242, input_243, input_244], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_58.run(buf265, primals_547, primals_548, primals_549, primals_550, primals_551, buf266, 4416, grid=grid(4416), stream=stream0)
        del primals_547
        # Topologically Sorted Source Nodes: [input_253], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf263, primals_570, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 545, 2, 2), (2180, 1, 1090, 545))
        # Topologically Sorted Source Nodes: [input_245], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_552, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 1656, 2, 2), (6624, 1, 3312, 1656))
        buf268 = buf267; del buf267  # reuse
        buf269 = empty_strided_cuda((4, 1656, 2, 2), (6624, 1, 3312, 1656), torch.float32)
        # Topologically Sorted Source Nodes: [input_245, input_246, input_247], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_59.run(buf268, primals_553, primals_554, primals_555, primals_556, primals_557, buf269, 26496, grid=grid(26496), stream=stream0)
        del primals_553
        # Topologically Sorted Source Nodes: [input_248], Original ATen: [aten.convolution]
        buf270 = extern_kernels.convolution(buf269, primals_558, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1656, bias=None)
        assert_size_stride(buf270, (4, 1656, 2, 2), (6624, 1, 3312, 1656))
        buf271 = buf270; del buf270  # reuse
        buf272 = empty_strided_cuda((4, 1656, 2, 2), (6624, 1, 3312, 1656), torch.float32)
        # Topologically Sorted Source Nodes: [input_248, input_249, input_250], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_59.run(buf271, primals_559, primals_560, primals_561, primals_562, primals_563, buf272, 26496, grid=grid(26496), stream=stream0)
        del primals_559
        # Topologically Sorted Source Nodes: [input_251], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf272, primals_564, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (4, 545, 2, 2), (2180, 1, 1090, 545))
        buf274 = buf273; del buf273  # reuse
        buf276 = buf275; del buf275  # reuse
        buf277 = empty_strided_cuda((4, 545, 2, 2), (2180, 1, 1090, 545), torch.float32)
        # Topologically Sorted Source Nodes: [input_251, input_252, input_253, input_254, out_15], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_60.run(buf274, buf276, primals_565, primals_571, primals_566, primals_567, primals_568, primals_569, primals_572, primals_573, primals_574, primals_575, buf277, 8720, grid=grid(8720), stream=stream0)
        del primals_565
        del primals_569
        del primals_571
        del primals_575
        # Topologically Sorted Source Nodes: [input_255], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_576, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 230, 2, 2), (920, 1, 460, 230))
        buf279 = buf278; del buf278  # reuse
        buf280 = empty_strided_cuda((4, 230, 2, 2), (920, 1, 460, 230), torch.float32)
        # Topologically Sorted Source Nodes: [input_255, input_256, input_257], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_61.run(buf279, primals_577, primals_578, primals_579, primals_580, primals_581, buf280, 3680, grid=grid(3680), stream=stream0)
        del primals_577
        # Topologically Sorted Source Nodes: [input_266], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf280, primals_600, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf289, (4, 489, 2, 2), (1956, 1, 978, 489))
        # Topologically Sorted Source Nodes: [input_258], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_582, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (4, 1380, 2, 2), (5520, 1, 2760, 1380))
        buf282 = buf281; del buf281  # reuse
        buf283 = empty_strided_cuda((4, 1380, 2, 2), (5520, 1, 2760, 1380), torch.float32)
        # Topologically Sorted Source Nodes: [input_258, input_259, input_260], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_62.run(buf282, primals_583, primals_584, primals_585, primals_586, primals_587, buf283, 22080, grid=grid(22080), stream=stream0)
        del primals_583
        # Topologically Sorted Source Nodes: [input_261], Original ATen: [aten.convolution]
        buf284 = extern_kernels.convolution(buf283, primals_588, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1380, bias=None)
        assert_size_stride(buf284, (4, 1380, 2, 2), (5520, 1, 2760, 1380))
        buf285 = buf284; del buf284  # reuse
        buf286 = empty_strided_cuda((4, 1380, 2, 2), (5520, 1, 2760, 1380), torch.float32)
        # Topologically Sorted Source Nodes: [input_261, input_262, input_263], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_62.run(buf285, primals_589, primals_590, primals_591, primals_592, primals_593, buf286, 22080, grid=grid(22080), stream=stream0)
        del primals_589
        # Topologically Sorted Source Nodes: [input_264], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_594, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (4, 489, 2, 2), (1956, 1, 978, 489))
        buf288 = buf287; del buf287  # reuse
        buf290 = buf289; del buf289  # reuse
        buf291 = empty_strided_cuda((4, 489, 2, 2), (1956, 1, 978, 489), torch.float32)
        # Topologically Sorted Source Nodes: [input_264, input_265, input_266, input_267, out_16], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_63.run(buf288, buf290, primals_595, primals_601, primals_596, primals_597, primals_598, primals_599, primals_602, primals_603, primals_604, primals_605, buf291, 7824, grid=grid(7824), stream=stream0)
        del primals_595
        del primals_599
        del primals_601
        del primals_605
        # Topologically Sorted Source Nodes: [input_268], Original ATen: [aten.convolution]
        buf292 = extern_kernels.convolution(buf291, primals_606, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (4, 213, 2, 2), (852, 1, 426, 213))
        buf293 = buf292; del buf292  # reuse
        buf294 = empty_strided_cuda((4, 213, 2, 2), (852, 1, 426, 213), torch.float32)
        # Topologically Sorted Source Nodes: [input_268, input_269, input_270], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_64.run(buf293, primals_607, primals_608, primals_609, primals_610, primals_611, buf294, 3408, grid=grid(3408), stream=stream0)
        del primals_607
        # Topologically Sorted Source Nodes: [input_279], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf291, primals_630, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf303, (4, 469, 2, 2), (1876, 1, 938, 469))
        # Topologically Sorted Source Nodes: [input_271], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (4, 1278, 2, 2), (5112, 1, 2556, 1278))
        buf296 = buf295; del buf295  # reuse
        buf297 = empty_strided_cuda((4, 1278, 2, 2), (5112, 1, 2556, 1278), torch.float32)
        # Topologically Sorted Source Nodes: [input_271, input_272, input_273], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_65.run(buf296, primals_613, primals_614, primals_615, primals_616, primals_617, buf297, 20448, grid=grid(20448), stream=stream0)
        del primals_613
        # Topologically Sorted Source Nodes: [input_274], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_618, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1278, bias=None)
        assert_size_stride(buf298, (4, 1278, 2, 2), (5112, 1, 2556, 1278))
        buf299 = buf298; del buf298  # reuse
        buf300 = empty_strided_cuda((4, 1278, 2, 2), (5112, 1, 2556, 1278), torch.float32)
        # Topologically Sorted Source Nodes: [input_274, input_275, input_276], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_65.run(buf299, primals_619, primals_620, primals_621, primals_622, primals_623, buf300, 20448, grid=grid(20448), stream=stream0)
        del primals_619
        # Topologically Sorted Source Nodes: [input_277], Original ATen: [aten.convolution]
        buf301 = extern_kernels.convolution(buf300, primals_624, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 469, 2, 2), (1876, 1, 938, 469))
        buf302 = buf301; del buf301  # reuse
        buf304 = buf303; del buf303  # reuse
        buf305 = empty_strided_cuda((4, 469, 2, 2), (1876, 1, 938, 469), torch.float32)
        # Topologically Sorted Source Nodes: [input_277, input_278, input_279, input_280, out_17], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_66.run(buf302, buf304, primals_625, primals_631, primals_626, primals_627, primals_628, primals_629, primals_632, primals_633, primals_634, primals_635, buf305, 7504, grid=grid(7504), stream=stream0)
        del primals_625
        del primals_629
        del primals_631
        del primals_635
        # Topologically Sorted Source Nodes: [input_281], Original ATen: [aten.convolution]
        buf306 = extern_kernels.convolution(buf305, primals_636, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf306, (4, 189, 2, 2), (756, 1, 378, 189))
        buf307 = buf306; del buf306  # reuse
        buf308 = empty_strided_cuda((4, 189, 2, 2), (756, 1, 378, 189), torch.float32)
        # Topologically Sorted Source Nodes: [input_281, input_282, input_283], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_67.run(buf307, primals_637, primals_638, primals_639, primals_640, primals_641, buf308, 3024, grid=grid(3024), stream=stream0)
        del primals_637
        # Topologically Sorted Source Nodes: [input_284], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, primals_642, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (4, 105, 2, 2), (420, 1, 210, 105))
        buf310 = buf309; del buf309  # reuse
        buf311 = empty_strided_cuda((4, 105, 2, 2), (420, 1, 210, 105), torch.float32)
        # Topologically Sorted Source Nodes: [input_284, input_285, input_286], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_68.run(buf310, primals_643, primals_644, primals_645, primals_646, primals_647, buf311, 1680, grid=grid(1680), stream=stream0)
        del primals_643
        buf317 = empty_strided_cuda((4, 105, 4, 4), (1696, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_287], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_69.run(buf312, buf314, buf311, buf315, buf316, buf317, 6720, grid=grid(6720), stream=stream0)
        # Topologically Sorted Source Nodes: [input_384], Original ATen: [aten.convolution]
        buf429 = extern_kernels.convolution(buf308, primals_864, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf429, (4, 462, 2, 2), (1848, 1, 924, 462))
        # Topologically Sorted Source Nodes: [input_376], Original ATen: [aten.convolution]
        buf421 = extern_kernels.convolution(buf308, primals_846, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf421, (4, 1134, 2, 2), (4536, 1, 2268, 1134))
        buf422 = buf421; del buf421  # reuse
        buf423 = empty_strided_cuda((4, 1134, 2, 2), (4536, 1, 2268, 1134), torch.float32)
        # Topologically Sorted Source Nodes: [input_376, input_377, input_378], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_70.run(buf422, primals_847, primals_848, primals_849, primals_850, primals_851, buf423, 18144, grid=grid(18144), stream=stream0)
        del primals_847
        # Topologically Sorted Source Nodes: [input_379], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_852, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1134, bias=None)
        assert_size_stride(buf424, (4, 1134, 2, 2), (4536, 1, 2268, 1134))
        buf425 = buf424; del buf424  # reuse
        buf426 = empty_strided_cuda((4, 1134, 2, 2), (4536, 1, 2268, 1134), torch.float32)
        # Topologically Sorted Source Nodes: [input_379, input_380, input_381], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_70.run(buf425, primals_853, primals_854, primals_855, primals_856, primals_857, buf426, 18144, grid=grid(18144), stream=stream0)
        del primals_853
        # Topologically Sorted Source Nodes: [input_382], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_858, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (4, 462, 2, 2), (1848, 1, 924, 462))
        buf428 = buf427; del buf427  # reuse
        buf430 = buf429; del buf429  # reuse
        buf431 = empty_strided_cuda((4, 462, 2, 2), (1848, 1, 924, 462), torch.float32)
        # Topologically Sorted Source Nodes: [input_382, input_383, input_384, input_385, out_24], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_71.run(buf428, buf430, primals_859, primals_865, primals_860, primals_861, primals_862, primals_863, primals_866, primals_867, primals_868, primals_869, buf431, 7392, grid=grid(7392), stream=stream0)
        del primals_859
        del primals_863
        del primals_865
        del primals_869
        # Topologically Sorted Source Nodes: [input_386], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_870, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (4, 75, 2, 2), (300, 1, 150, 75))
        buf433 = buf432; del buf432  # reuse
        # Topologically Sorted Source Nodes: [input_386], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_72.run(buf433, primals_871, 1200, grid=grid(1200), stream=stream0)
        del primals_871
        buf434 = empty_strided_cuda((4, 75, 2, 2), (300, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_387, input_388], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_73.run(buf433, primals_872, primals_873, primals_874, primals_875, buf434, 300, 4, grid=grid(300, 4), stream=stream0)
        buf319 = empty_strided_cuda((4, 430, 4, 4), (6880, 1, 1720, 430), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_74.run(buf254, buf317, buf313, buf314, buf311, buf315, buf316, buf318, buf319, 27520, grid=grid(27520), stream=stream0)
        del buf311
        del buf317
        # Topologically Sorted Source Nodes: [input_288], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_648, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 113, 4, 4), (1808, 1, 452, 113))
        buf321 = buf320; del buf320  # reuse
        buf322 = empty_strided_cuda((4, 113, 4, 4), (1808, 1, 452, 113), torch.float32)
        # Topologically Sorted Source Nodes: [input_288, input_289, input_290], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_75.run(buf321, primals_649, primals_650, primals_651, primals_652, primals_653, buf322, 7232, grid=grid(7232), stream=stream0)
        del primals_649
        # Topologically Sorted Source Nodes: [input_299], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf319, primals_672, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (4, 325, 4, 4), (5200, 1, 1300, 325))
        # Topologically Sorted Source Nodes: [input_291], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_654, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf323, (4, 678, 4, 4), (10848, 1, 2712, 678))
        buf324 = buf323; del buf323  # reuse
        buf325 = empty_strided_cuda((4, 678, 4, 4), (10848, 1, 2712, 678), torch.float32)
        # Topologically Sorted Source Nodes: [input_291, input_292, input_293], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_76.run(buf324, primals_655, primals_656, primals_657, primals_658, primals_659, buf325, 43392, grid=grid(43392), stream=stream0)
        del primals_655
        # Topologically Sorted Source Nodes: [input_294], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_660, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=678, bias=None)
        assert_size_stride(buf326, (4, 678, 4, 4), (10848, 1, 2712, 678))
        buf327 = buf326; del buf326  # reuse
        buf328 = empty_strided_cuda((4, 678, 4, 4), (10848, 1, 2712, 678), torch.float32)
        # Topologically Sorted Source Nodes: [input_294, input_295, input_296], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_76.run(buf327, primals_661, primals_662, primals_663, primals_664, primals_665, buf328, 43392, grid=grid(43392), stream=stream0)
        del primals_661
        # Topologically Sorted Source Nodes: [input_297], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_666, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (4, 325, 4, 4), (5200, 1, 1300, 325))
        buf330 = buf329; del buf329  # reuse
        buf332 = buf331; del buf331  # reuse
        buf333 = empty_strided_cuda((4, 325, 4, 4), (5200, 1, 1300, 325), torch.float32)
        # Topologically Sorted Source Nodes: [input_297, input_298, input_299, input_300, out_18], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_42.run(buf330, buf332, primals_667, primals_673, primals_668, primals_669, primals_670, primals_671, primals_674, primals_675, primals_676, primals_677, buf333, 20800, grid=grid(20800), stream=stream0)
        del primals_667
        del primals_671
        del primals_673
        del primals_677
        # Topologically Sorted Source Nodes: [input_301], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf333, primals_678, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 99, 4, 4), (1584, 1, 396, 99))
        buf335 = buf334; del buf334  # reuse
        buf336 = empty_strided_cuda((4, 99, 4, 4), (1584, 1, 396, 99), torch.float32)
        # Topologically Sorted Source Nodes: [input_301, input_302, input_303], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_77.run(buf335, primals_679, primals_680, primals_681, primals_682, primals_683, buf336, 6336, grid=grid(6336), stream=stream0)
        del primals_679
        # Topologically Sorted Source Nodes: [input_312], Original ATen: [aten.convolution]
        buf345 = extern_kernels.convolution(buf333, primals_702, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (4, 207, 4, 4), (3312, 1, 828, 207))
        # Topologically Sorted Source Nodes: [input_304], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(buf336, primals_684, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (4, 594, 4, 4), (9504, 1, 2376, 594))
        buf338 = buf337; del buf337  # reuse
        buf339 = empty_strided_cuda((4, 594, 4, 4), (9504, 1, 2376, 594), torch.float32)
        # Topologically Sorted Source Nodes: [input_304, input_305, input_306], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_78.run(buf338, primals_685, primals_686, primals_687, primals_688, primals_689, buf339, 38016, grid=grid(38016), stream=stream0)
        del primals_685
        # Topologically Sorted Source Nodes: [input_307], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_690, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=594, bias=None)
        assert_size_stride(buf340, (4, 594, 4, 4), (9504, 1, 2376, 594))
        buf341 = buf340; del buf340  # reuse
        buf342 = empty_strided_cuda((4, 594, 4, 4), (9504, 1, 2376, 594), torch.float32)
        # Topologically Sorted Source Nodes: [input_307, input_308, input_309], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_78.run(buf341, primals_691, primals_692, primals_693, primals_694, primals_695, buf342, 38016, grid=grid(38016), stream=stream0)
        del primals_691
        # Topologically Sorted Source Nodes: [input_310], Original ATen: [aten.convolution]
        buf343 = extern_kernels.convolution(buf342, primals_696, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf343, (4, 207, 4, 4), (3312, 1, 828, 207))
        buf344 = buf343; del buf343  # reuse
        buf346 = buf345; del buf345  # reuse
        buf347 = empty_strided_cuda((4, 207, 4, 4), (3312, 1, 828, 207), torch.float32)
        # Topologically Sorted Source Nodes: [input_310, input_311, input_312, input_313, out_19], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_79.run(buf344, buf346, primals_697, primals_703, primals_698, primals_699, primals_700, primals_701, primals_704, primals_705, primals_706, primals_707, buf347, 13248, grid=grid(13248), stream=stream0)
        del primals_697
        del primals_701
        del primals_703
        del primals_707
        # Topologically Sorted Source Nodes: [input_314], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, primals_708, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (4, 98, 4, 4), (1568, 1, 392, 98))
        buf349 = buf348; del buf348  # reuse
        buf350 = empty_strided_cuda((4, 98, 4, 4), (1568, 1, 392, 98), torch.float32)
        # Topologically Sorted Source Nodes: [input_314, input_315, input_316], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_80.run(buf349, primals_709, primals_710, primals_711, primals_712, primals_713, buf350, 6272, grid=grid(6272), stream=stream0)
        del primals_709
        # Topologically Sorted Source Nodes: [input_317], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_714, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (4, 47, 4, 4), (752, 1, 188, 47))
        buf352 = buf351; del buf351  # reuse
        buf353 = empty_strided_cuda((4, 47, 4, 4), (752, 1, 188, 47), torch.float32)
        # Topologically Sorted Source Nodes: [input_317, input_318, input_319], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_81.run(buf352, primals_715, primals_716, primals_717, primals_718, primals_719, buf353, 3008, grid=grid(3008), stream=stream0)
        del primals_715
        # Topologically Sorted Source Nodes: [input_371], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf350, primals_834, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (4, 183, 4, 4), (2928, 1, 732, 183))
        buf359 = empty_strided_cuda((4, 47, 8, 8), (3008, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_320], Original ATen: [aten._unsafe_index, aten.sub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_add_mul_sub_82.run(buf354, buf356, buf353, buf357, buf358, buf359, 12032, grid=grid(12032), stream=stream0)
        buf361 = empty_strided_cuda((4, 197, 8, 8), (12608, 1, 1576, 197), torch.float32)
        # Topologically Sorted Source Nodes: [cat_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_83.run(buf133, buf359, buf355, buf356, buf353, buf357, buf358, buf360, buf361, 50432, grid=grid(50432), stream=stream0)
        del buf353
        # Topologically Sorted Source Nodes: [input_321], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_720, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (4, 58, 8, 8), (3712, 1, 464, 58))
        buf363 = buf362; del buf362  # reuse
        buf364 = empty_strided_cuda((4, 58, 8, 8), (3712, 1, 464, 58), torch.float32)
        # Topologically Sorted Source Nodes: [input_321, input_322, input_323], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_84.run(buf363, primals_721, primals_722, primals_723, primals_724, primals_725, buf364, 14848, grid=grid(14848), stream=stream0)
        del primals_721
        # Topologically Sorted Source Nodes: [input_332], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf361, primals_744, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 122, 8, 8), (7808, 1, 976, 122))
        # Topologically Sorted Source Nodes: [input_363], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf350, primals_816, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (4, 588, 4, 4), (9408, 1, 2352, 588))
        buf408 = buf407; del buf407  # reuse
        buf409 = empty_strided_cuda((4, 588, 4, 4), (9408, 1, 2352, 588), torch.float32)
        # Topologically Sorted Source Nodes: [input_363, input_364, input_365], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_85.run(buf408, primals_817, primals_818, primals_819, primals_820, primals_821, buf409, 37632, grid=grid(37632), stream=stream0)
        del primals_817
        # Topologically Sorted Source Nodes: [input_366], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf409, primals_822, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=588, bias=None)
        assert_size_stride(buf410, (4, 588, 4, 4), (9408, 1, 2352, 588))
        buf411 = buf410; del buf410  # reuse
        buf412 = empty_strided_cuda((4, 588, 4, 4), (9408, 1, 2352, 588), torch.float32)
        # Topologically Sorted Source Nodes: [input_366, input_367, input_368], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_85.run(buf411, primals_823, primals_824, primals_825, primals_826, primals_827, buf412, 37632, grid=grid(37632), stream=stream0)
        del primals_823
        # Topologically Sorted Source Nodes: [input_369], Original ATen: [aten.convolution]
        buf413 = extern_kernels.convolution(buf412, primals_828, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf413, (4, 183, 4, 4), (2928, 1, 732, 183))
        buf414 = buf413; del buf413  # reuse
        buf416 = buf415; del buf415  # reuse
        buf417 = empty_strided_cuda((4, 183, 4, 4), (2928, 1, 732, 183), torch.float32)
        # Topologically Sorted Source Nodes: [input_369, input_370, input_371, input_372, out_23], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_86.run(buf414, buf416, primals_829, primals_835, primals_830, primals_831, primals_832, primals_833, primals_836, primals_837, primals_838, primals_839, buf417, 11712, grid=grid(11712), stream=stream0)
        del primals_829
        del primals_833
        del primals_835
        del primals_839
        # Topologically Sorted Source Nodes: [input_373], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, primals_840, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (4, 75, 4, 4), (1200, 1, 300, 75))
        buf419 = buf418; del buf418  # reuse
        # Topologically Sorted Source Nodes: [input_373], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_87.run(buf419, primals_841, 4800, grid=grid(4800), stream=stream0)
        del primals_841
        buf420 = empty_strided_cuda((4, 75, 4, 4), (1200, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_374, input_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_88.run(buf419, primals_842, primals_843, primals_844, primals_845, buf420, 300, 16, grid=grid(300, 16), stream=stream0)
        # Topologically Sorted Source Nodes: [input_324], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_726, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (4, 348, 8, 8), (22272, 1, 2784, 348))
        buf366 = buf365; del buf365  # reuse
        buf367 = empty_strided_cuda((4, 348, 8, 8), (22272, 1, 2784, 348), torch.float32)
        # Topologically Sorted Source Nodes: [input_324, input_325, input_326], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_89.run(buf366, primals_727, primals_728, primals_729, primals_730, primals_731, buf367, 89088, grid=grid(89088), stream=stream0)
        del primals_727
        # Topologically Sorted Source Nodes: [input_327], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf367, primals_732, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=348, bias=None)
        assert_size_stride(buf368, (4, 348, 8, 8), (22272, 1, 2784, 348))
        buf369 = buf368; del buf368  # reuse
        buf370 = empty_strided_cuda((4, 348, 8, 8), (22272, 1, 2784, 348), torch.float32)
        # Topologically Sorted Source Nodes: [input_327, input_328, input_329], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_89.run(buf369, primals_733, primals_734, primals_735, primals_736, primals_737, buf370, 89088, grid=grid(89088), stream=stream0)
        del primals_733
        # Topologically Sorted Source Nodes: [input_330], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_738, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (4, 122, 8, 8), (7808, 1, 976, 122))
        buf372 = buf371; del buf371  # reuse
        buf374 = buf373; del buf373  # reuse
        buf375 = empty_strided_cuda((4, 122, 8, 8), (7808, 1, 976, 122), torch.float32)
        # Topologically Sorted Source Nodes: [input_330, input_331, input_332, input_333, out_20], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_90.run(buf372, buf374, primals_739, primals_745, primals_740, primals_741, primals_742, primals_743, primals_746, primals_747, primals_748, primals_749, buf375, 31232, grid=grid(31232), stream=stream0)
        del primals_739
        del primals_743
        del primals_745
        del primals_749
        # Topologically Sorted Source Nodes: [input_334], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf375, primals_750, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (4, 52, 8, 8), (3328, 1, 416, 52))
        buf377 = buf376; del buf376  # reuse
        buf378 = empty_strided_cuda((4, 52, 8, 8), (3328, 1, 416, 52), torch.float32)
        # Topologically Sorted Source Nodes: [input_334, input_335, input_336], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_91.run(buf377, primals_751, primals_752, primals_753, primals_754, primals_755, buf378, 13312, grid=grid(13312), stream=stream0)
        del primals_751
        # Topologically Sorted Source Nodes: [input_345], Original ATen: [aten.convolution]
        buf387 = extern_kernels.convolution(buf375, primals_774, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (4, 87, 8, 8), (5568, 1, 696, 87))
        # Topologically Sorted Source Nodes: [input_337], Original ATen: [aten.convolution]
        buf379 = extern_kernels.convolution(buf378, primals_756, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf379, (4, 312, 8, 8), (19968, 1, 2496, 312))
        buf380 = buf379; del buf379  # reuse
        buf381 = empty_strided_cuda((4, 312, 8, 8), (19968, 1, 2496, 312), torch.float32)
        # Topologically Sorted Source Nodes: [input_337, input_338, input_339], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_92.run(buf380, primals_757, primals_758, primals_759, primals_760, primals_761, buf381, 79872, grid=grid(79872), stream=stream0)
        del primals_757
        # Topologically Sorted Source Nodes: [input_340], Original ATen: [aten.convolution]
        buf382 = extern_kernels.convolution(buf381, primals_762, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=312, bias=None)
        assert_size_stride(buf382, (4, 312, 8, 8), (19968, 1, 2496, 312))
        buf383 = buf382; del buf382  # reuse
        buf384 = empty_strided_cuda((4, 312, 8, 8), (19968, 1, 2496, 312), torch.float32)
        # Topologically Sorted Source Nodes: [input_340, input_341, input_342], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_92.run(buf383, primals_763, primals_764, primals_765, primals_766, primals_767, buf384, 79872, grid=grid(79872), stream=stream0)
        del primals_763
        # Topologically Sorted Source Nodes: [input_343], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_768, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (4, 87, 8, 8), (5568, 1, 696, 87))
        buf386 = buf385; del buf385  # reuse
        buf388 = buf387; del buf387  # reuse
        buf389 = empty_strided_cuda((4, 87, 8, 8), (5568, 1, 696, 87), torch.float32)
        # Topologically Sorted Source Nodes: [input_343, input_344, input_345, input_346, out_21], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_93.run(buf386, buf388, primals_769, primals_775, primals_770, primals_771, primals_772, primals_773, primals_776, primals_777, primals_778, primals_779, buf389, 22272, grid=grid(22272), stream=stream0)
        del primals_769
        del primals_773
        del primals_775
        del primals_779
        # Topologically Sorted Source Nodes: [input_347], Original ATen: [aten.convolution]
        buf390 = extern_kernels.convolution(buf389, primals_780, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (4, 47, 8, 8), (3008, 1, 376, 47))
        buf391 = buf390; del buf390  # reuse
        buf392 = reinterpret_tensor(buf359, (4, 47, 8, 8), (3008, 1, 376, 47), 0); del buf359  # reuse
        # Topologically Sorted Source Nodes: [input_347, input_348, input_349], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_94.run(buf391, primals_781, primals_782, primals_783, primals_784, primals_785, buf392, 12032, grid=grid(12032), stream=stream0)
        del primals_781
        # Topologically Sorted Source Nodes: [input_358], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf389, primals_804, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (4, 93, 8, 8), (5952, 1, 744, 93))
        # Topologically Sorted Source Nodes: [input_350], Original ATen: [aten.convolution]
        buf393 = extern_kernels.convolution(buf392, primals_786, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (4, 282, 8, 8), (18048, 1, 2256, 282))
        buf394 = buf393; del buf393  # reuse
        buf395 = empty_strided_cuda((4, 282, 8, 8), (18048, 1, 2256, 282), torch.float32)
        # Topologically Sorted Source Nodes: [input_350, input_351, input_352], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_95.run(buf394, primals_787, primals_788, primals_789, primals_790, primals_791, buf395, 72192, grid=grid(72192), stream=stream0)
        del primals_787
        # Topologically Sorted Source Nodes: [input_353], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_792, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=282, bias=None)
        assert_size_stride(buf396, (4, 282, 8, 8), (18048, 1, 2256, 282))
        buf397 = buf396; del buf396  # reuse
        buf398 = empty_strided_cuda((4, 282, 8, 8), (18048, 1, 2256, 282), torch.float32)
        # Topologically Sorted Source Nodes: [input_353, input_354, input_355], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_95.run(buf397, primals_793, primals_794, primals_795, primals_796, primals_797, buf398, 72192, grid=grid(72192), stream=stream0)
        del primals_793
        # Topologically Sorted Source Nodes: [input_356], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_798, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (4, 93, 8, 8), (5952, 1, 744, 93))
        buf400 = buf399; del buf399  # reuse
        buf402 = buf401; del buf401  # reuse
        buf403 = empty_strided_cuda((4, 93, 8, 8), (5952, 1, 744, 93), torch.float32)
        # Topologically Sorted Source Nodes: [input_356, input_357, input_358, input_359, out_22], Original ATen: [aten.convolution, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_convolution_96.run(buf400, buf402, primals_799, primals_805, primals_800, primals_801, primals_802, primals_803, primals_806, primals_807, primals_808, primals_809, buf403, 23808, grid=grid(23808), stream=stream0)
        del primals_799
        del primals_803
        del primals_805
        del primals_809
        # Topologically Sorted Source Nodes: [input_360], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_810, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 75, 8, 8), (4800, 1, 600, 75))
        buf405 = buf404; del buf404  # reuse
        # Topologically Sorted Source Nodes: [input_360], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_97.run(buf405, primals_811, 19200, grid=grid(19200), stream=stream0)
        del primals_811
        buf406 = empty_strided_cuda((4, 75, 8, 8), (4800, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_361, input_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_98.run(buf405, primals_812, primals_813, primals_814, primals_815, buf406, 300, 64, grid=grid(300, 64), stream=stream0)
    return (buf406, buf420, buf434, buf0, buf1, primals_4, primals_5, primals_6, primals_7, buf2, primals_10, primals_11, primals_12, primals_13, primals_14, primals_16, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_26, primals_28, primals_29, primals_30, primals_31, primals_32, primals_34, primals_35, primals_36, primals_38, primals_40, primals_41, primals_42, primals_44, primals_46, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_56, primals_58, primals_59, primals_60, primals_62, primals_64, primals_65, primals_66, primals_67, primals_68, primals_70, primals_71, primals_72, primals_73, primals_74, primals_76, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_86, primals_88, primals_89, primals_90, primals_92, primals_94, primals_95, primals_96, primals_97, primals_98, primals_100, primals_101, primals_102, primals_103, primals_104, primals_106, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_118, primals_119, primals_120, primals_122, primals_124, primals_125, primals_126, primals_127, primals_128, primals_130, primals_131, primals_132, primals_133, primals_134, primals_136, primals_137, primals_138, primals_140, primals_142, primals_143, primals_144, primals_145, primals_146, primals_148, primals_149, primals_150, primals_151, primals_152, primals_154, primals_155, primals_156, primals_157, primals_158, primals_160, primals_161, primals_162, primals_164, primals_166, primals_167, primals_168, primals_170, primals_172, primals_173, primals_174, primals_175, primals_180, primals_182, primals_183, primals_184, primals_185, primals_186, primals_188, primals_189, primals_190, primals_191, primals_192, primals_194, primals_195, primals_196, primals_197, primals_198, primals_200, primals_201, primals_202, primals_204, primals_206, primals_207, primals_208, primals_210, primals_212, primals_213, primals_214, primals_215, primals_216, primals_218, primals_219, primals_220, primals_221, primals_222, primals_224, primals_225, primals_226, primals_227, primals_228, primals_230, primals_231, primals_232, primals_234, primals_236, primals_237, primals_238, primals_240, primals_242, primals_243, primals_244, primals_245, primals_246, primals_248, primals_249, primals_250, primals_251, primals_252, primals_254, primals_255, primals_256, primals_257, primals_258, primals_260, primals_261, primals_262, primals_264, primals_266, primals_267, primals_268, primals_270, primals_272, primals_273, primals_274, primals_275, primals_276, primals_278, primals_279, primals_280, primals_281, primals_282, primals_284, primals_285, primals_286, primals_288, primals_290, primals_291, primals_292, primals_293, primals_294, primals_296, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_306, primals_308, primals_309, primals_310, primals_312, primals_314, primals_315, primals_316, primals_318, primals_320, primals_321, primals_322, primals_323, primals_324, primals_326, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_336, primals_338, primals_339, primals_340, primals_342, primals_344, primals_345, primals_346, primals_348, primals_350, primals_351, primals_352, primals_353, primals_354, primals_356, primals_357, primals_358, primals_359, primals_360, primals_362, primals_363, primals_364, primals_365, primals_366, primals_368, primals_369, primals_370, primals_372, primals_374, primals_375, primals_376, primals_378, primals_380, primals_381, primals_382, primals_383, primals_384, primals_386, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_396, primals_398, primals_399, primals_400, primals_402, primals_404, primals_405, primals_406, primals_408, primals_410, primals_411, primals_412, primals_413, primals_414, primals_416, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_426, primals_428, primals_429, primals_430, primals_432, primals_434, primals_435, primals_436, primals_438, primals_440, primals_441, primals_442, primals_443, primals_444, primals_446, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_456, primals_458, primals_459, primals_460, primals_462, primals_464, primals_465, primals_466, primals_468, primals_470, primals_471, primals_472, primals_473, primals_474, primals_476, primals_477, primals_478, primals_479, primals_480, primals_482, primals_483, primals_484, primals_485, primals_486, primals_488, primals_489, primals_490, primals_492, primals_494, primals_495, primals_496, primals_498, primals_500, primals_501, primals_502, primals_503, primals_504, primals_506, primals_507, primals_508, primals_509, primals_510, primals_512, primals_513, primals_514, primals_515, primals_516, primals_518, primals_519, primals_520, primals_522, primals_524, primals_525, primals_526, primals_528, primals_530, primals_531, primals_532, primals_533, primals_534, primals_536, primals_537, primals_538, primals_539, primals_540, primals_542, primals_543, primals_544, primals_546, primals_548, primals_549, primals_550, primals_551, primals_552, primals_554, primals_555, primals_556, primals_557, primals_558, primals_560, primals_561, primals_562, primals_563, primals_564, primals_566, primals_567, primals_568, primals_570, primals_572, primals_573, primals_574, primals_576, primals_578, primals_579, primals_580, primals_581, primals_582, primals_584, primals_585, primals_586, primals_587, primals_588, primals_590, primals_591, primals_592, primals_593, primals_594, primals_596, primals_597, primals_598, primals_600, primals_602, primals_603, primals_604, primals_606, primals_608, primals_609, primals_610, primals_611, primals_612, primals_614, primals_615, primals_616, primals_617, primals_618, primals_620, primals_621, primals_622, primals_623, primals_624, primals_626, primals_627, primals_628, primals_630, primals_632, primals_633, primals_634, primals_636, primals_638, primals_639, primals_640, primals_641, primals_642, primals_644, primals_645, primals_646, primals_647, primals_648, primals_650, primals_651, primals_652, primals_653, primals_654, primals_656, primals_657, primals_658, primals_659, primals_660, primals_662, primals_663, primals_664, primals_665, primals_666, primals_668, primals_669, primals_670, primals_672, primals_674, primals_675, primals_676, primals_678, primals_680, primals_681, primals_682, primals_683, primals_684, primals_686, primals_687, primals_688, primals_689, primals_690, primals_692, primals_693, primals_694, primals_695, primals_696, primals_698, primals_699, primals_700, primals_702, primals_704, primals_705, primals_706, primals_708, primals_710, primals_711, primals_712, primals_713, primals_714, primals_716, primals_717, primals_718, primals_719, primals_720, primals_722, primals_723, primals_724, primals_725, primals_726, primals_728, primals_729, primals_730, primals_731, primals_732, primals_734, primals_735, primals_736, primals_737, primals_738, primals_740, primals_741, primals_742, primals_744, primals_746, primals_747, primals_748, primals_750, primals_752, primals_753, primals_754, primals_755, primals_756, primals_758, primals_759, primals_760, primals_761, primals_762, primals_764, primals_765, primals_766, primals_767, primals_768, primals_770, primals_771, primals_772, primals_774, primals_776, primals_777, primals_778, primals_780, primals_782, primals_783, primals_784, primals_785, primals_786, primals_788, primals_789, primals_790, primals_791, primals_792, primals_794, primals_795, primals_796, primals_797, primals_798, primals_800, primals_801, primals_802, primals_804, primals_806, primals_807, primals_808, primals_810, primals_812, primals_813, primals_814, primals_815, primals_816, primals_818, primals_819, primals_820, primals_821, primals_822, primals_824, primals_825, primals_826, primals_827, primals_828, primals_830, primals_831, primals_832, primals_834, primals_836, primals_837, primals_838, primals_840, primals_842, primals_843, primals_844, primals_845, primals_846, primals_848, primals_849, primals_850, primals_851, primals_852, primals_854, primals_855, primals_856, primals_857, primals_858, primals_860, primals_861, primals_862, primals_864, primals_866, primals_867, primals_868, primals_870, primals_872, primals_873, primals_874, primals_875, buf4, buf5, buf7, buf8, buf10, buf11, buf13, buf14, buf16, buf17, buf19, buf21, buf22, buf24, buf25, buf27, buf28, buf30, buf31, buf33, buf34, buf36, buf37, buf39, buf40, buf42, buf44, buf45, buf47, buf48, buf50, buf51, buf53, buf54, buf56, buf58, buf59, buf61, buf62, buf64, buf65, buf67, buf68, buf70, buf71, buf73, buf74, buf76, buf77, buf79, buf81, buf82, buf84, reinterpret_tensor(buf87, (4, 150), (150, 1), 0), buf89, buf90, buf91, buf93, buf94, buf96, buf97, buf99, buf100, buf102, buf104, buf105, buf107, buf108, buf110, buf111, buf113, buf114, buf116, buf118, buf119, buf121, buf122, buf124, buf125, buf127, buf128, buf130, buf132, buf133, buf135, buf136, buf138, buf139, buf141, buf142, buf144, buf145, buf147, buf148, buf150, buf151, buf153, buf155, buf156, buf158, buf159, buf161, buf162, buf164, buf165, buf167, buf169, buf170, buf172, buf173, buf175, buf176, buf178, buf179, buf181, buf183, buf184, buf186, buf187, buf189, buf190, buf192, buf193, buf195, buf197, buf198, buf200, buf201, buf203, buf204, buf206, buf207, buf209, buf211, buf212, buf214, buf215, buf217, buf218, buf220, buf221, buf223, buf225, buf226, buf228, buf229, buf231, buf232, buf234, buf235, buf237, buf239, buf240, buf242, buf243, buf245, buf246, buf248, buf249, buf251, buf253, buf254, buf256, buf257, buf259, buf260, buf262, buf263, buf265, buf266, buf268, buf269, buf271, buf272, buf274, buf276, buf277, buf279, buf280, buf282, buf283, buf285, buf286, buf288, buf290, buf291, buf293, buf294, buf296, buf297, buf299, buf300, buf302, buf304, buf305, buf307, buf308, buf310, buf312, buf313, buf314, buf315, buf316, buf318, buf319, buf321, buf322, buf324, buf325, buf327, buf328, buf330, buf332, buf333, buf335, buf336, buf338, buf339, buf341, buf342, buf344, buf346, buf347, buf349, buf350, buf352, buf354, buf355, buf356, buf357, buf358, buf360, buf361, buf363, buf364, buf366, buf367, buf369, buf370, buf372, buf374, buf375, buf377, buf378, buf380, buf381, buf383, buf384, buf386, buf388, buf389, buf391, buf392, buf394, buf395, buf397, buf398, buf400, buf402, buf403, buf405, buf408, buf409, buf411, buf412, buf414, buf416, buf417, buf419, buf422, buf423, buf425, buf426, buf428, buf430, buf431, buf433, primals_178, primals_176, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((12, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((24, 12, 3, 3), (108, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((7, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((42, 7, 1, 1), (7, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((42, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((42, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((24, 42, 1, 1), (42, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((24, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((70, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((25, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((150, 25, 1, 1), (25, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((150, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((70, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((70, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((24, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((144, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((144, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((70, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((70, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((70, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((420, 70, 1, 1), (70, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((420, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((420, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((150, 420, 1, 1), (420, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((56, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((56, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((336, 56, 1, 1), (56, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((336, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((150, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((150, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((150, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((18, 150), (150, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((150, 18), (18, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((73, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((73, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((73, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((73, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((73, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((73, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((438, 73, 1, 1), (73, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((438, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((438, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((150, 438, 1, 1), (438, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((150, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((71, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((71, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((71, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((71, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((71, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((71, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((426, 71, 1, 1), (71, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((426, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((426, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((150, 426, 1, 1), (426, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((150, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((75, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((450, 75, 1, 1), (75, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((450, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((450, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((150, 450, 1, 1), (450, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((150, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((150, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((900, 150, 1, 1), (150, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((900, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((900, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((325, 900, 1, 1), (900, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((132, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((132, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((792, 132, 1, 1), (132, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((792, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((325, 792, 1, 1), (792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((124, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((124, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((744, 124, 1, 1), (124, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((744, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((744, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((325, 744, 1, 1), (744, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((141, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((141, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((141, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((141, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((141, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((141, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((846, 141, 1, 1), (141, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((846, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((846, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((325, 846, 1, 1), (846, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((140, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((325, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((137, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((137, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((137, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((137, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((137, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((137, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((822, 137, 1, 1), (137, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((822, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((822, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((325, 822, 1, 1), (822, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((135, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((135, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((135, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((135, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((135, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((135, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((810, 135, 1, 1), (135, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((810, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((810, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((325, 810, 1, 1), (810, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((133, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((133, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((133, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((133, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((133, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((133, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((798, 133, 1, 1), (133, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((798, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((798, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((325, 798, 1, 1), (798, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((140, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((140, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((840, 140, 1, 1), (140, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((840, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((840, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((325, 840, 1, 1), (840, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((325, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((1950, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((1950, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((1950, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((545, 1950, 1, 1), (1950, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((276, 545, 1, 1), (545, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((276, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((1656, 276, 1, 1), (276, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((1656, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((1656, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((545, 1656, 1, 1), (1656, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((545, 545, 1, 1), (545, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((545, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((230, 545, 1, 1), (545, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((230, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((230, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((230, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((230, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((230, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((1380, 230, 1, 1), (230, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((1380, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((1380, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((489, 1380, 1, 1), (1380, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((489, 230, 1, 1), (230, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((489, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((213, 489, 1, 1), (489, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((213, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((213, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((213, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((213, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((213, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((1278, 213, 1, 1), (213, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1278, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((1278, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((469, 1278, 1, 1), (1278, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((469, 489, 1, 1), (489, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((469, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((189, 469, 1, 1), (469, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((189, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((189, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((189, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((189, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((189, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((105, 189, 1, 1), (189, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((105, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((105, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((105, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((105, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((105, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((113, 430, 1, 1), (430, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((113, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((113, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((113, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((113, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((113, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((678, 113, 1, 1), (113, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((678, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((678, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((325, 678, 1, 1), (678, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((325, 430, 1, 1), (430, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((325, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((99, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((99, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((99, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((99, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((99, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((99, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((594, 99, 1, 1), (99, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((594, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((594, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((207, 594, 1, 1), (594, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((207, 325, 1, 1), (325, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((207, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((98, 207, 1, 1), (207, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((98, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((98, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((98, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((98, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((98, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((47, 98, 1, 1), (98, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((58, 197, 1, 1), (197, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((348, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_729 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_732 = rand_strided((348, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_735 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((348, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_738 = rand_strided((122, 348, 1, 1), (348, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_741 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_744 = rand_strided((122, 197, 1, 1), (197, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_747 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((122, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_750 = rand_strided((52, 122, 1, 1), (122, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_753 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((52, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_756 = rand_strided((312, 52, 1, 1), (52, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_759 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_762 = rand_strided((312, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_765 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((312, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_768 = rand_strided((87, 312, 1, 1), (312, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_771 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_774 = rand_strided((87, 122, 1, 1), (122, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_777 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((87, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_780 = rand_strided((47, 87, 1, 1), (87, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_783 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((47, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_786 = rand_strided((282, 47, 1, 1), (47, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_789 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_792 = rand_strided((282, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_795 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((282, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_798 = rand_strided((93, 282, 1, 1), (282, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_801 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_804 = rand_strided((93, 87, 1, 1), (87, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_807 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((93, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_810 = rand_strided((75, 93, 1, 1), (93, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_813 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_816 = rand_strided((588, 98, 1, 1), (98, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_819 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_822 = rand_strided((588, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_825 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((588, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_828 = rand_strided((183, 588, 1, 1), (588, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_831 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_834 = rand_strided((183, 98, 1, 1), (98, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_837 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((183, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_840 = rand_strided((75, 183, 1, 1), (183, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_843 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_846 = rand_strided((1134, 189, 1, 1), (189, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_849 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_852 = rand_strided((1134, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_855 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((1134, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_858 = rand_strided((462, 1134, 1, 1), (1134, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_861 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_864 = rand_strided((462, 189, 1, 1), (189, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_867 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((462, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_870 = rand_strided((75, 462, 1, 1), (462, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_873 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((75, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
