# AOT ID: ['20_forward']
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


# kernel path: inductor_cache/uz/cuzkcidriagnsde6rzvozus4lbwvql6qv6iypn2yw3es6pc4k64n.py
# Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   input_65 => add_84, add_85, convert_element_type_18, convert_element_type_19, iota_9, mul_70, mul_71
# Graph fragment:
#   %iota_9 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (8,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_9, 1), kwargs = {})
#   %add_84 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_70, 0), kwargs = {})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_84, torch.float32), kwargs = {})
#   %add_85 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_18, 0.0), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_85, 0.5), kwargs = {})
#   %convert_element_type_19 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_71, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_0 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/nb/cnbcfqjpdiilvumndk2w6vo2yvmlvsg27zbyu5menpnf22qy5vgl.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_5 => add_45, add_46, convert_element_type, convert_element_type_1, iota, mul_36, mul_37
# Graph fragment:
#   %iota : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (16,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota, 1), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_36, 0), kwargs = {})
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_45, torch.float32), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type, 0.0), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_46, 0.5), kwargs = {})
#   %convert_element_type_1 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_37, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_1 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yx/cyxp4flkraoe2nwwc7yy24cc2hmatbgjyllg5kg7udxav6b4o2uk.py
# Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_6 => add_57, add_58, convert_element_type_6, convert_element_type_7, iota_3, mul_46, mul_47
# Graph fragment:
#   %iota_3 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (32,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_3, 1), kwargs = {})
#   %add_57 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_46, 0), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_57, torch.float32), kwargs = {})
#   %add_58 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_6, 0.0), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_58, 0.5), kwargs = {})
#   %convert_element_type_7 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_47, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_2 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_2', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_2(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/km/ckmhvvsdfbg2ekmj675qtdakmgd3lc5p7uotni73uqiyunqnu2fg.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
# Source node to ATen node mapping:
#   x_7 => add_69, add_70, convert_element_type_12, convert_element_type_13, iota_6, mul_56, mul_57
# Graph fragment:
#   %iota_6 : [num_users=1] = call_function[target=torch.ops.prims.iota.default](args = (64,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%iota_6, 1), kwargs = {})
#   %add_69 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, 0), kwargs = {})
#   %convert_element_type_12 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_69, torch.float32), kwargs = {})
#   %add_70 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_12, 0.0), kwargs = {})
#   %mul_57 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_70, 0.5), kwargs = {})
#   %convert_element_type_13 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_57, torch.int64), kwargs = {})
triton_poi_fused__to_copy_add_arange_mul_3 = async_compile.triton('triton_poi_fused__to_copy_add_arange_mul_3', '''
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
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_arange_mul_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_arange_mul_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.int32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/p6/cp62fa47se5lxpcdkq6nl6ygukvrkjmjytktnrjgl4plgctz42ri.py
# Topologically Sorted Source Nodes: [x, input_1], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_1 => var_mean
#   x => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_3, %primals_1, %primals_2, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_4 = async_compile.triton('triton_red_fused_convolution_native_group_norm_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 65536},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_4(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x1 = ((xindex // 4) % 32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r3 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r3 + 65536*x4), xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r3 + 65536*x4), tmp2, xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp5, xmask)
    tl.store(out_ptr2 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xs/cxssycvft3zfkbnktz7qiexh7lyypduyxra4vf7q3og2nywjkmgw.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_1 => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_per_fused_native_group_norm_5 = async_compile.triton('triton_per_fused_native_group_norm_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 16*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1048576.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mi/cmif6imr2grtrsvwgf5ue3phiujyntedw6kkrco5ht2kbmzcoh4p.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_1, mul_1
#   input_2 => relu
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_7), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_3), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused_native_group_norm_relu_6 = async_compile.triton('triton_poi_fused_native_group_norm_relu_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 262144
    x1 = ((xindex // 262144) % 32)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 4), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1048576.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/kb/ckb64kf6tduiryfbsgq5cx7be2njaas67s6gs3xi6uwnq4lyhvqj.py
# Topologically Sorted Source Nodes: [y, y_1], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   y => convolution_2
#   y_1 => add_4
# Graph fragment:
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_1, %primals_10, %primals_11, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_2, %relu_1), kwargs = {})
triton_poi_fused_add_convolution_7 = async_compile.triton('triton_poi_fused_add_convolution_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33554432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/ky/cky6uxd3tkid7s35tutnn77oyehw5knic2g7fiw7wah5sfmtv572.py
# Topologically Sorted Source Nodes: [x_2, input_6], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_6 => var_mean_2
#   x_2 => convolution_3
# Graph fragment:
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_4, %primals_12, %primals_13, [2, 2, 2], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_8 = async_compile.triton('triton_red_fused_convolution_native_group_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_8(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x1 = ((xindex // 4) % 64)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r3 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r3 + 8192*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp5, xmask)
    tl.store(out_ptr2 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rv/crvvahhx3sgo7jodydwsxtgbctnkni4sljw6flfdediqsxnepzdo.py
# Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_6 => add_5, rsqrt_2, var_mean_2
# Graph fragment:
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_4, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
triton_per_fused_native_group_norm_9 = async_compile.triton('triton_per_fused_native_group_norm_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 32},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 32
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 32*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 32*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 262144.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/54/c54ad5dag2laqgk2mp6ow3g2p3zcurhue353plqjnd55tc662yeu.py
# Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_6, mul_5
#   input_7 => relu_2
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, %unsqueeze_23), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_19), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused_native_group_norm_relu_10 = async_compile.triton('triton_poi_fused_native_group_norm_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 32768
    x1 = ((xindex // 32768) % 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 8), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 8), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 262144.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/5q/c5qosr5d3twjda6kzudgbrl56uxsq77yaumqi7jm6dx53vczyuzk.py
# Topologically Sorted Source Nodes: [y_2, input_11], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_11 => var_mean_4
#   y_2 => convolution_5
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_3, %primals_20, %primals_21, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_8, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_11 = async_compile.triton('triton_red_fused_convolution_native_group_norm_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_11(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x1 = ((xindex // 4) % 64)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r3 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r3 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r3 + 8192*x4), tmp2, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pv/cpvspospspn4dt4snjgfysgxya6pkznbmrc7hhid5zqerakgm4wt.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_11 => add_11, mul_9
#   input_12 => relu_4
# Graph fragment:
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_9, %unsqueeze_39), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %unsqueeze_35), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused_native_group_norm_relu_12 = async_compile.triton('triton_poi_fused_native_group_norm_relu_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x6 = xindex // 32768
    x2 = ((xindex // 32768) % 64)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp3 = tl.load(in_ptr2 + (x6 // 8), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x6 // 8), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 262144.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/54/c54udw3i664spodkht25kxd4j4autc7uqorksdv77wlszos52l6t.py
# Topologically Sorted Source Nodes: [y_4, y_5], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   y_4 => convolution_7
#   y_5 => add_14
# Graph fragment:
#   %convolution_7 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_5, %primals_28, %primals_29, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_7, %relu_5), kwargs = {})
triton_poi_fused_add_convolution_13 = async_compile.triton('triton_poi_fused_add_convolution_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_13(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 32768) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/l2/cl2eurluoi6ddwerse433tns27l5xstyh2d7ougmvbzvq3umillt.py
# Topologically Sorted Source Nodes: [x_3, input_16], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_16 => var_mean_6
#   x_3 => convolution_8
# Graph fragment:
#   %convolution_8 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_14, %primals_30, %primals_31, [2, 2, 2], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_14 = async_compile.triton('triton_red_fused_convolution_native_group_norm_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_14(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 64)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp5, xmask)
    tl.store(out_ptr2 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/m5/cm5pyigh4qo3i6xyl5l5zfdhnpmz7zttudgmpdvtc2qsp2oskfzu.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_16 => add_15, rsqrt_6, var_mean_6
# Graph fragment:
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_12, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_6 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
triton_per_fused_native_group_norm_15 = async_compile.triton('triton_per_fused_native_group_norm_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 8},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 8*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 8*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 8*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 65536.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2w/c2waocmia4uup6fvqlxzkstn6j2pdqywgmzwyoddc7ygbwq3rkzr.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_16, mul_13
#   input_17 => relu_6
# Graph fragment:
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, %unsqueeze_55), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_13, %unsqueeze_51), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
triton_poi_fused_native_group_norm_relu_16 = async_compile.triton('triton_poi_fused_native_group_norm_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 4096
    x1 = ((xindex // 4096) % 128)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 16), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 16), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 65536.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/fk/cfkl6vdkcf3ofsavtpgyj65u5rfb6yhxwfqxpeguv4mjp5pxos6m.py
# Topologically Sorted Source Nodes: [y_6, input_21], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_21 => var_mean_8
#   y_6 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_7, %primals_38, %primals_39, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %var_mean_8 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_16, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_17 = async_compile.triton('triton_red_fused_convolution_native_group_norm_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_17(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 64)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 4096
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp2, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yy/cyyvxklnoonmanowdubiqakohplefxrexmxzps3bocvjm6hsrfor.py
# Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_21 => add_21, mul_17
#   input_22 => relu_8
# Graph fragment:
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_17, %unsqueeze_71), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_67), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
triton_poi_fused_native_group_norm_relu_18 = async_compile.triton('triton_poi_fused_native_group_norm_relu_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x6 = xindex // 4096
    x2 = ((xindex // 4096) % 128)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp3 = tl.load(in_ptr2 + (x6 // 16), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x6 // 16), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 65536.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/yy/cyytc2icjehrax3smloxurg27l7vnjyo5fwhgjxdye2jcfs5colv.py
# Topologically Sorted Source Nodes: [y_8, y_9], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   y_8 => convolution_12
#   y_9 => add_24
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_9, %primals_46, %primals_47, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_24 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %relu_9), kwargs = {})
triton_poi_fused_add_convolution_19 = async_compile.triton('triton_poi_fused_add_convolution_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_19', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_19(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 4096) % 128)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/fq/cfq5boqc2moxz2vrkvjqmsv442rk64fqj2egeoqmpatioksqx2f2.py
# Topologically Sorted Source Nodes: [x_4, input_26], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_26 => var_mean_10
#   x_4 => convolution_13
# Graph fragment:
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%add_24, %primals_48, %primals_49, [2, 2, 2], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_20, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_20 = async_compile.triton('triton_red_fused_convolution_native_group_norm_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_20(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 16)
    tmp4_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 512
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(rmask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(rmask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(rmask & xmask, tmp4_weight_next, tmp4_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp2, rmask & xmask)
    tmp4_tmp, tmp5_tmp, tmp6_tmp = triton_helpers.welford(
        tmp4_mean, tmp4_m2, tmp4_weight, 1
    )
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp4, xmask)
    tl.store(out_ptr1 + (x4), tmp5, xmask)
    tl.store(out_ptr2 + (x4), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/va/cvauy2s4armxrkpfi25fbgb6rzaaozj63ocni2doh3zsaljo7ow3.py
# Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   input_26 => add_25, rsqrt_10, var_mean_10
# Graph fragment:
#   %var_mean_10 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_20, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-05), kwargs = {})
#   %rsqrt_10 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_25,), kwargs = {})
triton_per_fused_native_group_norm_21 = async_compile.triton('triton_per_fused_native_group_norm_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32, 'r': 2},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 2*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 2*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + 2*x0), xmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp3, 0)
    tmp8 = tl.where(xmask, tmp4, 0)
    tmp9 = tl.where(xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 16384.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = libdevice.rsqrt(tmp19)
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a4/ca4r2na2exectkjncrxhibf7fs5chk4v324dczmgok5tjyp6iz4i.py
# Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_26 => add_26, mul_21
#   input_27 => relu_10
# Graph fragment:
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, %unsqueeze_87), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %unsqueeze_83), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
triton_poi_fused_native_group_norm_relu_22 = async_compile.triton('triton_poi_fused_native_group_norm_relu_22', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x4 = xindex // 512
    x1 = ((xindex // 512) % 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4 // 32), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4 // 32), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 16384.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full([1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp14, tmp13)
    tl.store(out_ptr0 + (x3), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/5x/c5xukqd23znpiu55th3pfk6qhwnjfhuqn3b4pdwdzme2jvyatsm3.py
# Topologically Sorted Source Nodes: [y_10, input_31], Original ATen: [aten.convolution, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_31 => var_mean_12
#   y_10 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_11, %primals_56, %primals_57, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_24, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_convolution_native_group_norm_23 = async_compile.triton('triton_red_fused_convolution_native_group_norm_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_native_group_norm_23', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_23(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 16)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 512
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp2, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/my/cmylqgzu6l6tdx34kdlsgf5mqn2ypaucpwt6h6gue7uuuvzszw7h.py
# Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.native_group_norm, aten.relu]
# Source node to ATen node mapping:
#   input_31 => add_31, mul_25
#   input_32 => relu_12
# Graph fragment:
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_25, %unsqueeze_103), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_25, %unsqueeze_99), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused_native_group_norm_relu_24 = async_compile.triton('triton_poi_fused_native_group_norm_relu_24', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_group_norm_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_group_norm_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x4 = xindex
    x6 = xindex // 512
    x2 = ((xindex // 512) % 256)
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x4), None)
    tmp3 = tl.load(in_ptr2 + (x6 // 32), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x6 // 32), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 16384.0
    tmp7 = tmp5 / tmp6
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x4), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/zp/czp47m4ox5ae5ph72w3oiy3bxd7elsjbg4cmqfo3t6zlmuk5fovj.py
# Topologically Sorted Source Nodes: [y_16, y_17, input_61], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   input_61 => var_mean_24
#   y_16 => convolution_21
#   y_17 => add_44
# Graph fragment:
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_17, %primals_80, %primals_81, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_21, %relu_17), kwargs = {})
#   %var_mean_24 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_48, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused_add_convolution_native_group_norm_25 = async_compile.triton('triton_red_fused_add_convolution_native_group_norm_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 64, 'r': 8192},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_native_group_norm_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_native_group_norm_25(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x4 = xindex
    x0 = (xindex % 16)
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex
        r3 = rindex // 512
        tmp0 = tl.load(in_out_ptr0 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (r3 + 16*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r5 + 8192*x4), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
        tl.store(in_out_ptr0 + (r5 + 8192*x4), tmp4, rmask & xmask)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tn/ctnhk2r467eoeoptdlq6oaxcbntq3dkpsvrynoq6gnxejh5fsqdg.py
# Topologically Sorted Source Nodes: [conv3d_22, x_5, add_9, input_46], Original ATen: [aten.convolution, aten._unsafe_index, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_9 => add_51
#   conv3d_22 => convolution_22
#   input_46 => var_mean_18
#   x_5 => _unsafe_index
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_44, %primals_82, %primals_83, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_unsafe_index : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_22, [None, None, %unsqueeze_145, %unsqueeze_144, %convert_element_type_1]), kwargs = {})
#   %add_51 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index, %add_24), kwargs = {})
#   %var_mean_18 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_36, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__unsafe_index_add_convolution_native_group_norm_26 = async_compile.triton('triton_red_fused__unsafe_index_add_convolution_native_group_norm_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_index_add_convolution_native_group_norm_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_index_add_convolution_native_group_norm_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x6 = xindex
    x0 = (xindex % 64)
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = ((rindex // 256) % 16)
        r3 = ((rindex // 16) % 16)
        r2 = (rindex % 16)
        r5 = rindex // 4096
        r7 = rindex
        tmp0 = tl.load(in_ptr0 + (r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r5 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r7 + 8192*x6), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 8, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = tl.load(in_ptr1 + (tmp12 + 8*tmp8 + 64*tmp4 + 512*r5 + 1024*x6), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight, roffset == 0
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
        tl.store(out_ptr0 + (r7 + 8192*x6), tmp17, rmask & xmask)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr1 + (x6), tmp19, xmask)
    tl.store(out_ptr2 + (x6), tmp20, xmask)
    tl.store(out_ptr3 + (x6), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/y5/cy55xqcq2qa2qcnu2dphaj5o32h4ahy2v3nxgcyzmubetz5hw7ig.py
# Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_63 => convolution_32
# Graph fragment:
#   %convolution_32 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_24, %primals_116, %primals_117, [2, 2, 2], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_27 = async_compile.triton('triton_poi_fused_convolution_27', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_27(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 64) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/og/cognnqun67wugr6n5n2v65l2zimou2co2xeigus27hgt45bksqpa.py
# Topologically Sorted Source Nodes: [mul, std, mul_1, y_25], Original ATen: [aten.mul, aten.exp, aten.add]
# Source node to ATen node mapping:
#   mul => mul_68
#   mul_1 => mul_69
#   std => exp
#   y_25 => add_83
# Graph fragment:
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%slice_4, 0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_68,), kwargs = {})
#   %mul_69 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%randn, %exp), kwargs = {})
#   %add_83 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_69, %slice_2), kwargs = {})
triton_poi_fused_add_exp_mul_28 = async_compile.triton('triton_poi_fused_add_exp_mul_28', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_mul_28', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_mul_28(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    x1 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (128 + x0 + 256*x1), xmask)
    tmp6 = tl.load(in_ptr1 + (x0 + 256*x1), xmask)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp5 = tmp0 * tmp4
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x2), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xa/cxaffloxbm7giohhjvvdqymyb2py5ezdfigaeseycuma33pqgvx6.py
# Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_64 => relu_25
# Graph fragment:
#   %relu_25 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_51,), kwargs = {})
triton_poi_fused_relu_29 = async_compile.triton('triton_poi_fused_relu_29', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_29', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_29(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/d4/cd4gvyman5dwfbk4soc3cke3u75ixltatfrhok63eu7hzt7u4iy6.py
# Topologically Sorted Source Nodes: [conv3d_33, input_65], Original ATen: [aten.convolution, aten._unsafe_index]
# Source node to ATen node mapping:
#   conv3d_33 => convolution_33
#   input_65 => _unsafe_index_3
# Graph fragment:
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%relu_25, %primals_122, %primals_123, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_unsafe_index_3 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_33, [None, None, %unsqueeze_210, %unsqueeze_209, %convert_element_type_19]), kwargs = {})
triton_poi_fused__unsafe_index_convolution_30 = async_compile.triton('triton_poi_fused__unsafe_index_convolution_30', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_convolution_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__unsafe_index_convolution_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 64) % 8)
    x1 = ((xindex // 8) % 8)
    x0 = (xindex % 8)
    x6 = xindex // 512
    x3 = ((xindex // 512) % 256)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr2 + (x3), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tmp6 = tmp5 + tmp1
    tmp7 = tmp5 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp5)
    tmp10 = tmp9 + tmp1
    tmp11 = tmp9 < 0
    tmp12 = tl.where(tmp11, tmp10, tmp9)
    tmp13 = tl.load(in_ptr1 + (tmp12 + 4*tmp8 + 16*tmp4 + 64*x6), None, eviction_policy='evict_last')
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x7), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/7s/c7s45lcof43l7rbbulwegbqpn3eleshfszmtr63cujzw7jwk3sha.py
# Topologically Sorted Source Nodes: [conv3d_34, y_28, input_66], Original ATen: [aten.convolution, aten._unsafe_index, aten.native_group_norm]
# Source node to ATen node mapping:
#   conv3d_34 => convolution_34
#   input_66 => var_mean_25
#   y_28 => _unsafe_index_4
# Graph fragment:
#   %convolution_34 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index_3, %primals_124, %primals_125, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_unsafe_index_4 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_34, [None, None, %unsqueeze_145, %unsqueeze_144, %convert_element_type_1]), kwargs = {})
#   %var_mean_25 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_52, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__unsafe_index_convolution_native_group_norm_31 = async_compile.triton('triton_red_fused__unsafe_index_convolution_native_group_norm_31', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_index_convolution_native_group_norm_31', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_index_convolution_native_group_norm_31(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x6 = xindex
    x0 = (xindex % 64)
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r4 = ((rindex // 256) % 16)
        r3 = ((rindex // 16) % 16)
        r2 = (rindex % 16)
        r5 = rindex // 4096
        r7 = rindex
        tmp0 = tl.load(in_ptr0 + (r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr2 + (r5 + 2*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 8, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = tl.load(in_ptr1 + (tmp12 + 8*tmp8 + 64*tmp4 + 512*r5 + 1024*x6), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp15 = tmp13 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr0 + (r7 + 8192*x6), tmp15, rmask & xmask)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr1 + (x6), tmp17, xmask)
    tl.store(out_ptr2 + (x6), tmp18, xmask)
    tl.store(out_ptr3 + (x6), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ev/cevs6i6q26spr7hsyvqpkfcs2drivefnrj7vzvsvq37xezghna4p.py
# Topologically Sorted Source Nodes: [conv3d_25, x_6, add_11, input_51], Original ATen: [aten.convolution, aten._unsafe_index, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_11 => add_63
#   conv3d_25 => convolution_25
#   input_51 => var_mean_20
#   x_6 => _unsafe_index_1
# Graph fragment:
#   %convolution_25 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_56, %primals_92, %primals_93, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_unsafe_index_1 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_25, [None, None, %unsqueeze_164, %unsqueeze_163, %convert_element_type_7]), kwargs = {})
#   %add_63 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_1, %add_14), kwargs = {})
#   %var_mean_20 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_40, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__unsafe_index_add_convolution_native_group_norm_32 = async_compile.triton('triton_red_fused__unsafe_index_add_convolution_native_group_norm_32', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_index_add_convolution_native_group_norm_32', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_index_add_convolution_native_group_norm_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x8 = xindex // 4
    x1 = ((xindex // 4) % 64)
    tmp14 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    x6 = xindex
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex // 1024
        r4 = ((rindex // 32) % 32)
        r3 = (rindex % 32)
        r7 = rindex
        tmp0 = tl.load(in_ptr0 + (r5 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr3 + (r7 + 8192*x6), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 16, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = tl.load(in_ptr1 + (tmp12 + 16*tmp8 + 256*tmp4 + 4096*x8), rmask & xmask, eviction_policy='evict_last')
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight, roffset == 0
        )
        tmp19_mean = tl.where(rmask & xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(rmask & xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(rmask & xmask, tmp19_weight_next, tmp19_weight)
        tl.store(out_ptr0 + (r7 + 8192*x6), tmp17, rmask & xmask)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr1 + (x6), tmp19, xmask)
    tl.store(out_ptr2 + (x6), tmp20, xmask)
    tl.store(out_ptr3 + (x6), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvufye4t4qgcnv7awu2oaqubrgkts4trqohebzwjbrqt5mgtqpsa.py
# Topologically Sorted Source Nodes: [conv3d_28, x_7, add_13, input_56], Original ATen: [aten.convolution, aten._unsafe_index, aten.add, aten.native_group_norm]
# Source node to ATen node mapping:
#   add_13 => add_75
#   conv3d_28 => convolution_28
#   input_56 => var_mean_22
#   x_7 => _unsafe_index_2
# Graph fragment:
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_68, %primals_102, %primals_103, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_unsafe_index_2 : [num_users=1] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_28, [None, None, %unsqueeze_183, %unsqueeze_182, %convert_element_type_13]), kwargs = {})
#   %add_75 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%_unsafe_index_2, %add_4), kwargs = {})
#   %var_mean_22 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_44, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__unsafe_index_add_convolution_native_group_norm_33 = async_compile.triton('triton_red_fused__unsafe_index_add_convolution_native_group_norm_33', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 65536},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_index_add_convolution_native_group_norm_33', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_index_add_convolution_native_group_norm_33(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x8 = xindex // 4
    x1 = ((xindex // 4) % 32)
    tmp14 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    x6 = xindex
    tmp19_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp19_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex // 4096
        r4 = ((rindex // 64) % 64)
        r3 = (rindex % 64)
        r7 = rindex
        tmp0 = tl.load(in_ptr0 + (r5 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r4), None, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr0 + (r3), None, eviction_policy='evict_last')
        tmp16 = tl.load(in_ptr3 + (r7 + 65536*x6), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 32, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = tl.load(in_ptr1 + (tmp12 + 32*tmp8 + 1024*tmp4 + 32768*x8), xmask, eviction_policy='evict_last')
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
        tmp19_mean_next, tmp19_m2_next, tmp19_weight_next = triton_helpers.welford_reduce(
            tmp18, tmp19_mean, tmp19_m2, tmp19_weight, roffset == 0
        )
        tmp19_mean = tl.where(xmask, tmp19_mean_next, tmp19_mean)
        tmp19_m2 = tl.where(xmask, tmp19_m2_next, tmp19_m2)
        tmp19_weight = tl.where(xmask, tmp19_weight_next, tmp19_weight)
        tl.store(out_ptr0 + (r7 + 65536*x6), tmp17, xmask)
    tmp19_tmp, tmp20_tmp, tmp21_tmp = triton_helpers.welford(
        tmp19_mean, tmp19_m2, tmp19_weight, 1
    )
    tmp19 = tmp19_tmp[:, None]
    tmp20 = tmp20_tmp[:, None]
    tmp21 = tmp21_tmp[:, None]
    tl.store(out_ptr1 + (x6), tmp19, xmask)
    tl.store(out_ptr2 + (x6), tmp20, xmask)
    tl.store(out_ptr3 + (x6), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mc/cmcwq5454sc4cy26h22jxv63jeve26ii2nsznlktuvwx74oxdjtk.py
# Topologically Sorted Source Nodes: [conv3d_37, y_31, input_71], Original ATen: [aten.convolution, aten._unsafe_index, aten.native_group_norm]
# Source node to ATen node mapping:
#   conv3d_37 => convolution_37
#   input_71 => var_mean_27
#   y_31 => _unsafe_index_5
# Graph fragment:
#   %convolution_37 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_100, %primals_134, %primals_135, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_unsafe_index_5 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_37, [None, None, %unsqueeze_164, %unsqueeze_163, %convert_element_type_7]), kwargs = {})
#   %var_mean_27 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_56, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__unsafe_index_convolution_native_group_norm_34 = async_compile.triton('triton_red_fused__unsafe_index_convolution_native_group_norm_34', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1024, 'r': 8192},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_index_convolution_native_group_norm_34', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_index_convolution_native_group_norm_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x8 = xindex // 4
    x1 = ((xindex // 4) % 64)
    tmp14 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    x6 = xindex
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r5 = rindex // 1024
        r4 = ((rindex // 32) % 32)
        r3 = (rindex % 32)
        r7 = rindex
        tmp0 = tl.load(in_ptr0 + (r5 + 8*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r4), rmask, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr0 + (r3), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([XBLOCK, RBLOCK], 16, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = tl.load(in_ptr1 + (tmp12 + 16*tmp8 + 256*tmp4 + 4096*x8), rmask & xmask, eviction_policy='evict_last')
        tmp15 = tmp13 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(rmask & xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(rmask & xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(rmask & xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr0 + (r7 + 8192*x6), tmp15, rmask & xmask)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr1 + (x6), tmp17, xmask)
    tl.store(out_ptr2 + (x6), tmp18, xmask)
    tl.store(out_ptr3 + (x6), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/be/cbelpqb2wwwhh4dgyzhxx5epdiqibzv5qxcdkt26kklbj5vnzj7q.py
# Topologically Sorted Source Nodes: [conv3d_40, y_34, input_76], Original ATen: [aten.convolution, aten._unsafe_index, aten.native_group_norm]
# Source node to ATen node mapping:
#   conv3d_40 => convolution_40
#   input_76 => var_mean_29
#   y_34 => _unsafe_index_6
# Graph fragment:
#   %convolution_40 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_111, %primals_144, %primals_145, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %_unsafe_index_6 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%convolution_40, [None, None, %unsqueeze_183, %unsqueeze_182, %convert_element_type_13]), kwargs = {})
#   %var_mean_29 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view_60, [2, 3]), kwargs = {correction: 0, keepdim: True})
triton_red_fused__unsafe_index_convolution_native_group_norm_35 = async_compile.triton('triton_red_fused__unsafe_index_convolution_native_group_norm_35', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 512, 'r': 65536},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__unsafe_index_convolution_native_group_norm_35', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 3, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__unsafe_index_convolution_native_group_norm_35(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x8 = xindex // 4
    x1 = ((xindex // 4) % 32)
    tmp14 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    x6 = xindex
    tmp17_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
        r5 = rindex // 4096
        r4 = ((rindex // 64) % 64)
        r3 = (rindex % 64)
        r7 = rindex
        tmp0 = tl.load(in_ptr0 + (r5 + 16*x0), xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (r4), None, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr0 + (r3), None, eviction_policy='evict_last')
        tmp1 = tl.full([XBLOCK, RBLOCK], 32, tl.int32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp0 < 0
        tmp4 = tl.where(tmp3, tmp2, tmp0)
        tmp6 = tmp5 + tmp1
        tmp7 = tmp5 < 0
        tmp8 = tl.where(tmp7, tmp6, tmp5)
        tmp10 = tmp9 + tmp1
        tmp11 = tmp9 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp9)
        tmp13 = tl.load(in_ptr1 + (tmp12 + 32*tmp8 + 1024*tmp4 + 32768*x8), xmask, eviction_policy='evict_last')
        tmp15 = tmp13 + tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp17_mean_next, tmp17_m2_next, tmp17_weight_next = triton_helpers.welford_reduce(
            tmp16, tmp17_mean, tmp17_m2, tmp17_weight, roffset == 0
        )
        tmp17_mean = tl.where(xmask, tmp17_mean_next, tmp17_mean)
        tmp17_m2 = tl.where(xmask, tmp17_m2_next, tmp17_m2)
        tmp17_weight = tl.where(xmask, tmp17_weight_next, tmp17_weight)
        tl.store(out_ptr0 + (r7 + 65536*x6), tmp15, xmask)
    tmp17_tmp, tmp18_tmp, tmp19_tmp = triton_helpers.welford(
        tmp17_mean, tmp17_m2, tmp17_weight, 1
    )
    tmp17 = tmp17_tmp[:, None]
    tmp18 = tmp18_tmp[:, None]
    tmp19 = tmp19_tmp[:, None]
    tl.store(out_ptr1 + (x6), tmp17, xmask)
    tl.store(out_ptr2 + (x6), tmp18, xmask)
    tl.store(out_ptr3 + (x6), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5i/c5iuvfx2qmyiujmsicydopchk4og56lb5sis3uac5pbq4sxk6a2l.py
# Topologically Sorted Source Nodes: [dec], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   dec => convolution_43
# Graph fragment:
#   %convolution_43 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_122, %primals_154, %primals_155, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_36 = async_compile.triton('triton_poi_fused_convolution_36', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 2)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/v7/cv747t72linv64j2bet6jzpk7r67wam7vw3iaxmtdhyst5fr6rak.py
# Topologically Sorted Source Nodes: [y_24], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   y_24 => convolution_31
# Graph fragment:
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%add_80, %primals_112, %primals_113, [1, 1, 1], [1, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_37 = async_compile.triton('triton_poi_fused_convolution_37', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_37', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_37(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x1 = ((xindex // 262144) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155 = args
    args.clear()
    assert_size_stride(primals_1, (32, 2, 3, 3, 3), (54, 27, 9, 3, 1))
    assert_size_stride(primals_2, (32, ), (1, ))
    assert_size_stride(primals_3, (4, 2, 64, 64, 64), (524288, 262144, 4096, 64, 1))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (64, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_17, (64, ), (1, ))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (64, ), (1, ))
    assert_size_stride(primals_28, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (128, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (256, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_49, (256, ), (1, ))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_77, (256, ), (1, ))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (128, ), (1, ))
    assert_size_stride(primals_85, (128, ), (1, ))
    assert_size_stride(primals_86, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_87, (128, ), (1, ))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_91, (128, ), (1, ))
    assert_size_stride(primals_92, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, ), (1, ))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_97, (64, ), (1, ))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_101, (64, ), (1, ))
    assert_size_stride(primals_102, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_103, (32, ), (1, ))
    assert_size_stride(primals_104, (32, ), (1, ))
    assert_size_stride(primals_105, (32, ), (1, ))
    assert_size_stride(primals_106, (32, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_107, (32, ), (1, ))
    assert_size_stride(primals_108, (32, ), (1, ))
    assert_size_stride(primals_109, (32, ), (1, ))
    assert_size_stride(primals_110, (32, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_111, (32, ), (1, ))
    assert_size_stride(primals_112, (4, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_113, (4, ), (1, ))
    assert_size_stride(primals_114, (256, ), (1, ))
    assert_size_stride(primals_115, (256, ), (1, ))
    assert_size_stride(primals_116, (16, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(primals_117, (16, ), (1, ))
    assert_size_stride(primals_118, (256, 1024), (1024, 1))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (1024, 128), (128, 1))
    assert_size_stride(primals_121, (1024, ), (1, ))
    assert_size_stride(primals_122, (256, 16, 1, 1, 1), (16, 1, 1, 1, 1))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (128, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (64, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(primals_135, (64, ), (1, ))
    assert_size_stride(primals_136, (64, ), (1, ))
    assert_size_stride(primals_137, (64, ), (1, ))
    assert_size_stride(primals_138, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_139, (64, ), (1, ))
    assert_size_stride(primals_140, (64, ), (1, ))
    assert_size_stride(primals_141, (64, ), (1, ))
    assert_size_stride(primals_142, (64, 64, 3, 3, 3), (1728, 27, 9, 3, 1))
    assert_size_stride(primals_143, (64, ), (1, ))
    assert_size_stride(primals_144, (32, 64, 1, 1, 1), (64, 1, 1, 1, 1))
    assert_size_stride(primals_145, (32, ), (1, ))
    assert_size_stride(primals_146, (32, ), (1, ))
    assert_size_stride(primals_147, (32, ), (1, ))
    assert_size_stride(primals_148, (32, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_149, (32, ), (1, ))
    assert_size_stride(primals_150, (32, ), (1, ))
    assert_size_stride(primals_151, (32, ), (1, ))
    assert_size_stride(primals_152, (32, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_153, (32, ), (1, ))
    assert_size_stride(primals_154, (2, 32, 3, 3, 3), (864, 27, 9, 3, 1))
    assert_size_stride(primals_155, (2, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [eps], Original ATen: [aten.randn_like]
        buf270 = torch.ops.aten.randn.default([4, 128], dtype=torch.float32, device=device(type='cuda', index=0), pin_memory=False)
        buf271 = buf270
        del buf270
        buf276 = empty_strided_cuda((8, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [input_65], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_0.run(buf276, 8, grid=grid(8), stream=stream0)
        buf189 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_1.run(buf189, 16, grid=grid(16), stream=stream0)
        buf212 = empty_strided_cuda((32, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_2.run(buf212, 32, grid=grid(32), stream=stream0)
        buf235 = empty_strided_cuda((64, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.arange, aten.add, aten.mul, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_arange_mul_3.run(buf235, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((4, 8, 1, 1, 16), (128, 16, 512, 512, 1), torch.float32)
        buf3 = empty_strided_cuda((4, 8, 1, 1, 16), (128, 16, 512, 512, 1), torch.float32)
        buf4 = empty_strided_cuda((4, 8, 1, 1, 16), (128, 16, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x, input_1], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf1, primals_2, buf2, buf3, buf4, 512, 65536, grid=grid(512), stream=stream0)
        del primals_2
        buf5 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf6 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf8 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf2, buf3, buf4, buf5, buf6, buf8, 32, 16, grid=grid(32), stream=stream0)
        buf9 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_6.run(buf1, buf5, buf6, primals_4, primals_5, buf9, 33554432, grid=grid(33554432), stream=stream0)
        del primals_5
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_6, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf11 = buf10; del buf10  # reuse
        buf12 = buf4; del buf4  # reuse
        buf13 = buf3; del buf3  # reuse
        buf14 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf11, primals_7, buf12, buf13, buf14, 512, 65536, grid=grid(512), stream=stream0)
        del primals_7
        buf15 = buf6; del buf6  # reuse
        buf16 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf18 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf12, buf13, buf14, buf15, buf16, buf18, 32, 16, grid=grid(32), stream=stream0)
        buf19 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, input_5], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_6.run(buf11, buf15, buf16, primals_8, primals_9, buf19, 33554432, grid=grid(33554432), stream=stream0)
        del primals_9
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_10, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf21 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [y, y_1], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_7.run(buf21, primals_11, buf19, 33554432, grid=grid(33554432), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_12, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf23 = buf22; del buf22  # reuse
        buf24 = empty_strided_cuda((4, 8, 1, 1, 32), (256, 32, 1024, 1024, 1), torch.float32)
        buf25 = empty_strided_cuda((4, 8, 1, 1, 32), (256, 32, 1024, 1024, 1), torch.float32)
        buf26 = empty_strided_cuda((4, 8, 1, 1, 32), (256, 32, 1024, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_2, input_6], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_8.run(buf23, primals_13, buf24, buf25, buf26, 1024, 8192, grid=grid(1024), stream=stream0)
        del primals_13
        buf27 = buf16; del buf16  # reuse
        buf28 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf30 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf24, buf25, buf26, buf27, buf28, buf30, 32, 32, grid=grid(32), stream=stream0)
        buf31 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf23, buf27, buf28, primals_14, primals_15, buf31, 8388608, grid=grid(8388608), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_16, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf33 = buf32; del buf32  # reuse
        buf34 = buf26; del buf26  # reuse
        buf35 = buf25; del buf25  # reuse
        buf36 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_8.run(buf33, primals_17, buf34, buf35, buf36, 1024, 8192, grid=grid(1024), stream=stream0)
        del primals_17
        buf37 = buf28; del buf28  # reuse
        buf38 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf40 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf34, buf35, buf36, buf37, buf38, buf40, 32, 32, grid=grid(32), stream=stream0)
        buf41 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, input_10], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf33, buf37, buf38, primals_18, primals_19, buf41, 8388608, grid=grid(8388608), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [y_2], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_20, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf43 = buf42; del buf42  # reuse
        buf44 = buf36; del buf36  # reuse
        buf45 = buf35; del buf35  # reuse
        buf46 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [y_2, input_11], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_11.run(buf43, primals_21, buf41, buf44, buf45, buf46, 1024, 8192, grid=grid(1024), stream=stream0)
        del primals_21
        buf47 = buf38; del buf38  # reuse
        buf48 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf50 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf44, buf45, buf46, buf47, buf48, buf50, 32, 32, grid=grid(32), stream=stream0)
        buf51 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_12.run(buf43, buf41, buf47, buf48, primals_22, primals_23, buf51, 8388608, grid=grid(8388608), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_24, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf53 = buf52; del buf52  # reuse
        buf54 = buf46; del buf46  # reuse
        buf55 = buf45; del buf45  # reuse
        buf56 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_8.run(buf53, primals_25, buf54, buf55, buf56, 1024, 8192, grid=grid(1024), stream=stream0)
        del primals_25
        buf57 = buf48; del buf48  # reuse
        buf58 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf60 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf54, buf55, buf56, buf57, buf58, buf60, 32, 32, grid=grid(32), stream=stream0)
        buf61 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf53, buf57, buf58, primals_26, primals_27, buf61, 8388608, grid=grid(8388608), stream=stream0)
        del primals_27
        # Topologically Sorted Source Nodes: [y_4], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_28, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf63 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [y_4, y_5], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_13.run(buf63, primals_29, buf61, 8388608, grid=grid(8388608), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_30, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf65 = buf64; del buf64  # reuse
        buf66 = empty_strided_cuda((4, 8, 1, 1, 8), (64, 8, 256, 256, 1), torch.float32)
        buf67 = empty_strided_cuda((4, 8, 1, 1, 8), (64, 8, 256, 256, 1), torch.float32)
        buf68 = empty_strided_cuda((4, 8, 1, 1, 8), (64, 8, 256, 256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, input_16], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_14.run(buf65, primals_31, buf66, buf67, buf68, 256, 8192, grid=grid(256), stream=stream0)
        del primals_31
        buf69 = buf58; del buf58  # reuse
        buf70 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf72 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf66, buf67, buf68, buf69, buf70, buf72, 32, 8, grid=grid(32), stream=stream0)
        buf73 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf65, buf69, buf70, primals_32, primals_33, buf73, 2097152, grid=grid(2097152), stream=stream0)
        del primals_33
        # Topologically Sorted Source Nodes: [input_18], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_34, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf75 = buf74; del buf74  # reuse
        buf76 = buf68; del buf68  # reuse
        buf77 = buf67; del buf67  # reuse
        buf78 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_14.run(buf75, primals_35, buf76, buf77, buf78, 256, 8192, grid=grid(256), stream=stream0)
        del primals_35
        buf79 = buf70; del buf70  # reuse
        buf80 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf82 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf76, buf77, buf78, buf79, buf80, buf82, 32, 8, grid=grid(32), stream=stream0)
        buf83 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19, input_20], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf75, buf79, buf80, primals_36, primals_37, buf83, 2097152, grid=grid(2097152), stream=stream0)
        del primals_37
        # Topologically Sorted Source Nodes: [y_6], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_38, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf85 = buf84; del buf84  # reuse
        buf86 = buf78; del buf78  # reuse
        buf87 = buf77; del buf77  # reuse
        buf88 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [y_6, input_21], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_17.run(buf85, primals_39, buf83, buf86, buf87, buf88, 256, 8192, grid=grid(256), stream=stream0)
        del primals_39
        buf89 = buf80; del buf80  # reuse
        buf90 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf92 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_21], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf86, buf87, buf88, buf89, buf90, buf92, 32, 8, grid=grid(32), stream=stream0)
        buf93 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_18.run(buf85, buf83, buf89, buf90, primals_40, primals_41, buf93, 2097152, grid=grid(2097152), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_42, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf95 = buf94; del buf94  # reuse
        buf96 = buf88; del buf88  # reuse
        buf97 = buf87; del buf87  # reuse
        buf98 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [input_23, input_24], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_14.run(buf95, primals_43, buf96, buf97, buf98, 256, 8192, grid=grid(256), stream=stream0)
        del primals_43
        buf99 = buf90; del buf90  # reuse
        buf100 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf102 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_24], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf96, buf97, buf98, buf99, buf100, buf102, 32, 8, grid=grid(32), stream=stream0)
        buf103 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf95, buf99, buf100, primals_44, primals_45, buf103, 2097152, grid=grid(2097152), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [y_8], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_46, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf105 = buf104; del buf104  # reuse
        # Topologically Sorted Source Nodes: [y_8, y_9], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_19.run(buf105, primals_47, buf103, 2097152, grid=grid(2097152), stream=stream0)
        del primals_47
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_48, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf107 = buf106; del buf106  # reuse
        buf108 = empty_strided_cuda((4, 8, 1, 1, 2), (16, 2, 64, 64, 1), torch.float32)
        buf109 = empty_strided_cuda((4, 8, 1, 1, 2), (16, 2, 64, 64, 1), torch.float32)
        buf110 = empty_strided_cuda((4, 8, 1, 1, 2), (16, 2, 64, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, input_26], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_20.run(buf107, primals_49, buf108, buf109, buf110, 64, 8192, grid=grid(64), stream=stream0)
        del primals_49
        buf111 = buf100; del buf100  # reuse
        buf112 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf114 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf108, buf109, buf110, buf111, buf112, buf114, 32, 2, grid=grid(32), stream=stream0)
        buf115 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf107, buf111, buf112, primals_50, primals_51, buf115, 524288, grid=grid(524288), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [input_28], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_52, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf117 = buf116; del buf116  # reuse
        buf118 = buf110; del buf110  # reuse
        buf119 = buf109; del buf109  # reuse
        buf120 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_20.run(buf117, primals_53, buf118, buf119, buf120, 64, 8192, grid=grid(64), stream=stream0)
        del primals_53
        buf121 = buf112; del buf112  # reuse
        buf122 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf124 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf118, buf119, buf120, buf121, buf122, buf124, 32, 2, grid=grid(32), stream=stream0)
        buf125 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_29, input_30], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf117, buf121, buf122, primals_54, primals_55, buf125, 524288, grid=grid(524288), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [y_10], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_56, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf127 = buf126; del buf126  # reuse
        buf128 = buf120; del buf120  # reuse
        buf129 = buf119; del buf119  # reuse
        buf130 = buf118; del buf118  # reuse
        # Topologically Sorted Source Nodes: [y_10, input_31], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_23.run(buf127, primals_57, buf125, buf128, buf129, buf130, 64, 8192, grid=grid(64), stream=stream0)
        del primals_57
        buf131 = buf122; del buf122  # reuse
        buf132 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf134 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_31], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf128, buf129, buf130, buf131, buf132, buf134, 32, 2, grid=grid(32), stream=stream0)
        buf135 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_31, input_32], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf127, buf125, buf131, buf132, primals_58, primals_59, buf135, 524288, grid=grid(524288), stream=stream0)
        del primals_59
        # Topologically Sorted Source Nodes: [input_33], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_60, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf137 = buf136; del buf136  # reuse
        buf138 = buf130; del buf130  # reuse
        buf139 = buf129; del buf129  # reuse
        buf140 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_20.run(buf137, primals_61, buf138, buf139, buf140, 64, 8192, grid=grid(64), stream=stream0)
        del primals_61
        buf141 = buf132; del buf132  # reuse
        buf142 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf144 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_34], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf138, buf139, buf140, buf141, buf142, buf144, 32, 2, grid=grid(32), stream=stream0)
        buf145 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf137, buf141, buf142, primals_62, primals_63, buf145, 524288, grid=grid(524288), stream=stream0)
        del primals_63
        # Topologically Sorted Source Nodes: [y_12], Original ATen: [aten.convolution]
        buf146 = extern_kernels.convolution(buf145, primals_64, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf147 = buf146; del buf146  # reuse
        buf148 = buf140; del buf140  # reuse
        buf149 = buf139; del buf139  # reuse
        buf150 = buf138; del buf138  # reuse
        # Topologically Sorted Source Nodes: [y_12, input_36], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_23.run(buf147, primals_65, buf145, buf148, buf149, buf150, 64, 8192, grid=grid(64), stream=stream0)
        del primals_65
        buf151 = buf142; del buf142  # reuse
        buf152 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf154 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_36], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf148, buf149, buf150, buf151, buf152, buf154, 32, 2, grid=grid(32), stream=stream0)
        buf155 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf147, buf145, buf151, buf152, primals_66, primals_67, buf155, 524288, grid=grid(524288), stream=stream0)
        del primals_67
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_68, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf157 = buf156; del buf156  # reuse
        buf158 = buf150; del buf150  # reuse
        buf159 = buf149; del buf149  # reuse
        buf160 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_20.run(buf157, primals_69, buf158, buf159, buf160, 64, 8192, grid=grid(64), stream=stream0)
        del primals_69
        buf161 = buf152; del buf152  # reuse
        buf162 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf164 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_39], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf158, buf159, buf160, buf161, buf162, buf164, 32, 2, grid=grid(32), stream=stream0)
        buf165 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf157, buf161, buf162, primals_70, primals_71, buf165, 524288, grid=grid(524288), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [y_14], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_72, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf167 = buf166; del buf166  # reuse
        buf168 = buf160; del buf160  # reuse
        buf169 = buf159; del buf159  # reuse
        buf170 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [y_14, input_41], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_23.run(buf167, primals_73, buf165, buf168, buf169, buf170, 64, 8192, grid=grid(64), stream=stream0)
        del primals_73
        buf171 = buf162; del buf162  # reuse
        buf172 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf174 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf168, buf169, buf170, buf171, buf172, buf174, 32, 2, grid=grid(32), stream=stream0)
        buf175 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_41, input_42], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_24.run(buf167, buf165, buf171, buf172, primals_74, primals_75, buf175, 524288, grid=grid(524288), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [input_43], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, primals_76, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf177 = buf176; del buf176  # reuse
        buf178 = buf170; del buf170  # reuse
        buf179 = buf169; del buf169  # reuse
        buf180 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [input_43, input_44], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_20.run(buf177, primals_77, buf178, buf179, buf180, 64, 8192, grid=grid(64), stream=stream0)
        del primals_77
        buf181 = buf172; del buf172  # reuse
        buf182 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf184 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf178, buf179, buf180, buf181, buf182, buf184, 32, 2, grid=grid(32), stream=stream0)
        buf185 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf177, buf181, buf182, primals_78, primals_79, buf185, 524288, grid=grid(524288), stream=stream0)
        del primals_79
        # Topologically Sorted Source Nodes: [y_16], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, primals_80, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (4, 256, 8, 8, 8), (131072, 512, 64, 8, 1))
        buf187 = buf186; del buf186  # reuse
        buf259 = buf180; del buf180  # reuse
        buf260 = buf179; del buf179  # reuse
        buf261 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [y_16, y_17, input_61], Original ATen: [aten.convolution, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_native_group_norm_25.run(buf187, primals_81, buf185, buf259, buf260, buf261, 64, 8192, grid=grid(64), stream=stream0)
        del primals_81
        buf262 = buf182; del buf182  # reuse
        buf263 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf265 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_21.run(buf259, buf260, buf261, buf262, buf263, buf265, 32, 2, grid=grid(32), stream=stream0)
        del buf259
        del buf260
        del buf261
        # Topologically Sorted Source Nodes: [conv3d_22], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_82, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf190 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        buf191 = buf98; del buf98  # reuse
        buf192 = buf97; del buf97  # reuse
        buf193 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [conv3d_22, x_5, add_9, input_46], Original ATen: [aten.convolution, aten._unsafe_index, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_index_add_convolution_native_group_norm_26.run(buf189, buf188, primals_83, buf105, buf190, buf191, buf192, buf193, 256, 8192, grid=grid(256), stream=stream0)
        del buf188
        del primals_83
        buf194 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf195 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf197 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_46], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf191, buf192, buf193, buf194, buf195, buf197, 32, 8, grid=grid(32), stream=stream0)
        buf198 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, input_47], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf190, buf194, buf195, primals_84, primals_85, buf198, 2097152, grid=grid(2097152), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [input_48], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_86, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf200 = buf199; del buf199  # reuse
        buf201 = buf193; del buf193  # reuse
        buf202 = buf192; del buf192  # reuse
        buf203 = buf191; del buf191  # reuse
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_14.run(buf200, primals_87, buf201, buf202, buf203, 256, 8192, grid=grid(256), stream=stream0)
        del primals_87
        buf204 = buf195; del buf195  # reuse
        buf205 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf207 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_49], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf201, buf202, buf203, buf204, buf205, buf207, 32, 8, grid=grid(32), stream=stream0)
        buf266 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_61, input_62], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_22.run(buf187, buf262, buf263, primals_114, primals_115, buf266, 524288, grid=grid(524288), stream=stream0)
        del primals_115
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        buf267 = extern_kernels.convolution(buf266, primals_116, stride=(2, 2, 2), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf267, (4, 16, 4, 4, 4), (1024, 64, 16, 4, 1))
        buf268 = buf267; del buf267  # reuse
        # Topologically Sorted Source Nodes: [input_63], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_27.run(buf268, primals_117, 4096, grid=grid(4096), stream=stream0)
        del primals_117
        buf269 = reinterpret_tensor(buf56, (4, 256), (256, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_119, reinterpret_tensor(buf268, (4, 1024), (1024, 1), 0), reinterpret_tensor(primals_118, (1024, 256), (1, 1024), 0), alpha=1, beta=1, out=buf269)
        del primals_119
        buf272 = reinterpret_tensor(buf14, (4, 128), (128, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [mul, std, mul_1, y_25], Original ATen: [aten.mul, aten.exp, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_mul_28.run(buf271, buf269, buf272, 512, grid=grid(512), stream=stream0)
        buf273 = empty_strided_cuda((4, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [y_26], Original ATen: [aten.addmm]
        extern_kernels.mm(buf272, reinterpret_tensor(primals_120, (128, 1024), (1, 128), 0), out=buf273)
        buf274 = reinterpret_tensor(buf273, (4, 16, 4, 4, 4), (1024, 64, 16, 4, 1), 0); del buf273  # reuse
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_29.run(buf274, primals_121, 4096, grid=grid(4096), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [conv3d_33], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_122, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (4, 256, 4, 4, 4), (16384, 64, 16, 4, 1))
        buf277 = empty_strided_cuda((4, 256, 8, 8, 8), (131072, 512, 64, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv3d_33, input_65], Original ATen: [aten.convolution, aten._unsafe_index]
        stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_index_convolution_30.run(buf276, buf275, primals_123, buf277, 524288, grid=grid(524288), stream=stream0)
        del buf275
        del primals_123
        # Topologically Sorted Source Nodes: [conv3d_34], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_124, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 128, 8, 8, 8), (65536, 512, 64, 8, 1))
        buf279 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        buf280 = buf203; del buf203  # reuse
        buf281 = buf202; del buf202  # reuse
        buf282 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [conv3d_34, y_28, input_66], Original ATen: [aten.convolution, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_index_convolution_native_group_norm_31.run(buf189, buf278, primals_125, buf279, buf280, buf281, buf282, 256, 8192, grid=grid(256), stream=stream0)
        del buf278
        del primals_125
        buf283 = buf263; del buf263  # reuse
        buf284 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf286 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf280, buf281, buf282, buf283, buf284, buf286, 32, 8, grid=grid(32), stream=stream0)
        buf287 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf279, buf283, buf284, primals_126, primals_127, buf287, 2097152, grid=grid(2097152), stream=stream0)
        del primals_127
        # Topologically Sorted Source Nodes: [input_68], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_128, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf289 = buf288; del buf288  # reuse
        buf290 = buf282; del buf282  # reuse
        buf291 = buf281; del buf281  # reuse
        buf292 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_14.run(buf289, primals_129, buf290, buf291, buf292, 256, 8192, grid=grid(256), stream=stream0)
        del primals_129
        buf293 = buf284; del buf284  # reuse
        buf294 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf296 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_69], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_15.run(buf290, buf291, buf292, buf293, buf294, buf296, 32, 8, grid=grid(32), stream=stream0)
        del buf290
        del buf291
        del buf292
        buf208 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_49, input_50], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf200, buf204, buf205, primals_88, primals_89, buf208, 2097152, grid=grid(2097152), stream=stream0)
        del primals_89
        # Topologically Sorted Source Nodes: [y_18], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_90, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf210 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [y_18, y_19], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_19.run(buf210, primals_91, buf208, 2097152, grid=grid(2097152), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [conv3d_25], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_92, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf213 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        buf214 = buf55; del buf55  # reuse
        buf215 = buf54; del buf54  # reuse
        buf216 = empty_strided_cuda((4, 8, 1, 1, 32), (256, 32, 1024, 1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv3d_25, x_6, add_11, input_51], Original ATen: [aten.convolution, aten._unsafe_index, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_index_add_convolution_native_group_norm_32.run(buf212, buf211, primals_93, buf63, buf213, buf214, buf215, buf216, 1024, 8192, grid=grid(1024), stream=stream0)
        del buf211
        del primals_93
        buf217 = buf205; del buf205  # reuse
        buf218 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf220 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_51], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf214, buf215, buf216, buf217, buf218, buf220, 32, 32, grid=grid(32), stream=stream0)
        buf221 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf213, buf217, buf218, primals_94, primals_95, buf221, 8388608, grid=grid(8388608), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_96, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf223 = buf222; del buf222  # reuse
        buf224 = buf216; del buf216  # reuse
        buf225 = buf215; del buf215  # reuse
        buf226 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [input_53, input_54], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_8.run(buf223, primals_97, buf224, buf225, buf226, 1024, 8192, grid=grid(1024), stream=stream0)
        del primals_97
        buf227 = buf218; del buf218  # reuse
        buf228 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf230 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_54], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf224, buf225, buf226, buf227, buf228, buf230, 32, 32, grid=grid(32), stream=stream0)
        buf231 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf223, buf227, buf228, primals_98, primals_99, buf231, 8388608, grid=grid(8388608), stream=stream0)
        del primals_99
        # Topologically Sorted Source Nodes: [y_20], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, primals_100, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf233 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [y_20, y_21], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_13.run(buf233, primals_101, buf231, 8388608, grid=grid(8388608), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [conv3d_28], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_102, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1))
        buf236 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        buf237 = buf13; del buf13  # reuse
        buf238 = buf12; del buf12  # reuse
        buf239 = empty_strided_cuda((4, 8, 1, 1, 16), (128, 16, 512, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [conv3d_28, x_7, add_13, input_56], Original ATen: [aten.convolution, aten._unsafe_index, aten.add, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_index_add_convolution_native_group_norm_33.run(buf235, buf234, primals_103, buf21, buf236, buf237, buf238, buf239, 512, 65536, grid=grid(512), stream=stream0)
        del buf234
        del primals_103
        buf240 = buf228; del buf228  # reuse
        buf241 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf243 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf237, buf238, buf239, buf240, buf241, buf243, 32, 16, grid=grid(32), stream=stream0)
        buf244 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_6.run(buf236, buf240, buf241, primals_104, primals_105, buf244, 33554432, grid=grid(33554432), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [input_58], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_106, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf246 = buf245; del buf245  # reuse
        buf247 = buf239; del buf239  # reuse
        buf248 = buf238; del buf238  # reuse
        buf249 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [input_58, input_59], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf246, primals_107, buf247, buf248, buf249, 512, 65536, grid=grid(512), stream=stream0)
        del primals_107
        buf250 = buf241; del buf241  # reuse
        buf251 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf253 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf247, buf248, buf249, buf250, buf251, buf253, 32, 16, grid=grid(32), stream=stream0)
        buf297 = empty_strided_cuda((4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_69, input_70], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_16.run(buf289, buf293, buf294, primals_130, primals_131, buf297, 2097152, grid=grid(2097152), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [y_29], Original ATen: [aten.convolution]
        buf298 = extern_kernels.convolution(buf297, primals_132, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf298, (4, 128, 16, 16, 16), (524288, 4096, 256, 16, 1))
        buf299 = buf298; del buf298  # reuse
        # Topologically Sorted Source Nodes: [y_29, y_30], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_19.run(buf299, primals_133, buf297, 2097152, grid=grid(2097152), stream=stream0)
        del primals_133
        # Topologically Sorted Source Nodes: [conv3d_37], Original ATen: [aten.convolution]
        buf300 = extern_kernels.convolution(buf299, primals_134, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (4, 64, 16, 16, 16), (262144, 4096, 256, 16, 1))
        buf301 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        buf302 = buf226; del buf226  # reuse
        buf303 = buf225; del buf225  # reuse
        buf304 = buf224; del buf224  # reuse
        # Topologically Sorted Source Nodes: [conv3d_37, y_31, input_71], Original ATen: [aten.convolution, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_index_convolution_native_group_norm_34.run(buf212, buf300, primals_135, buf301, buf302, buf303, buf304, 1024, 8192, grid=grid(1024), stream=stream0)
        del buf300
        del primals_135
        buf305 = buf294; del buf294  # reuse
        buf306 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf308 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_71], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf302, buf303, buf304, buf305, buf306, buf308, 32, 32, grid=grid(32), stream=stream0)
        buf309 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf301, buf305, buf306, primals_136, primals_137, buf309, 8388608, grid=grid(8388608), stream=stream0)
        del primals_137
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf310 = extern_kernels.convolution(buf309, primals_138, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf311 = buf310; del buf310  # reuse
        buf312 = buf304; del buf304  # reuse
        buf313 = buf303; del buf303  # reuse
        buf314 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [input_73, input_74], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_8.run(buf311, primals_139, buf312, buf313, buf314, 1024, 8192, grid=grid(1024), stream=stream0)
        del primals_139
        buf315 = buf306; del buf306  # reuse
        buf316 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf318 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_74], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_9.run(buf312, buf313, buf314, buf315, buf316, buf318, 32, 32, grid=grid(32), stream=stream0)
        del buf312
        del buf313
        del buf314
        buf319 = empty_strided_cuda((4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_74, input_75], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_10.run(buf311, buf315, buf316, primals_140, primals_141, buf319, 8388608, grid=grid(8388608), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [y_32], Original ATen: [aten.convolution]
        buf320 = extern_kernels.convolution(buf319, primals_142, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf320, (4, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1))
        buf321 = buf320; del buf320  # reuse
        # Topologically Sorted Source Nodes: [y_32, y_33], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_13.run(buf321, primals_143, buf319, 8388608, grid=grid(8388608), stream=stream0)
        del primals_143
        # Topologically Sorted Source Nodes: [conv3d_40], Original ATen: [aten.convolution]
        buf322 = extern_kernels.convolution(buf321, primals_144, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf322, (4, 32, 32, 32, 32), (1048576, 32768, 1024, 32, 1))
        buf323 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        buf324 = buf249; del buf249  # reuse
        buf325 = buf248; del buf248  # reuse
        buf326 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [conv3d_40, y_34, input_76], Original ATen: [aten.convolution, aten._unsafe_index, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__unsafe_index_convolution_native_group_norm_35.run(buf235, buf322, primals_145, buf323, buf324, buf325, buf326, 512, 65536, grid=grid(512), stream=stream0)
        del buf322
        del primals_145
        buf327 = buf316; del buf316  # reuse
        buf328 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf330 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf324, buf325, buf326, buf327, buf328, buf330, 32, 16, grid=grid(32), stream=stream0)
        buf331 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, input_77], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_6.run(buf323, buf327, buf328, primals_146, primals_147, buf331, 33554432, grid=grid(33554432), stream=stream0)
        del primals_147
        # Topologically Sorted Source Nodes: [input_78], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_148, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf333 = buf332; del buf332  # reuse
        buf334 = buf326; del buf326  # reuse
        buf335 = buf325; del buf325  # reuse
        buf336 = buf324; del buf324  # reuse
        # Topologically Sorted Source Nodes: [input_78, input_79], Original ATen: [aten.convolution, aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_native_group_norm_4.run(buf333, primals_149, buf334, buf335, buf336, 512, 65536, grid=grid(512), stream=stream0)
        del primals_149
        buf337 = buf328; del buf328  # reuse
        buf338 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        buf340 = empty_strided_cuda((4, 8, 1, 1), (8, 1, 32, 32), torch.float32)
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_5.run(buf334, buf335, buf336, buf337, buf338, buf340, 32, 16, grid=grid(32), stream=stream0)
        del buf334
        del buf335
        del buf336
        buf341 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_79, input_80], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_6.run(buf333, buf337, buf338, primals_150, primals_151, buf341, 33554432, grid=grid(33554432), stream=stream0)
        del buf338
        del primals_151
        # Topologically Sorted Source Nodes: [y_35], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_152, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf343 = buf342; del buf342  # reuse
        # Topologically Sorted Source Nodes: [y_35, y_36], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_7.run(buf343, primals_153, buf341, 33554432, grid=grid(33554432), stream=stream0)
        del primals_153
        # Topologically Sorted Source Nodes: [dec], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_154, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 2, 64, 64, 64), (524288, 262144, 4096, 64, 1))
        buf345 = buf344; del buf344  # reuse
        # Topologically Sorted Source Nodes: [dec], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_36.run(buf345, primals_155, 2097152, grid=grid(2097152), stream=stream0)
        del primals_155
        buf254 = empty_strided_cuda((4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_59, input_60], Original ATen: [aten.native_group_norm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_group_norm_relu_6.run(buf246, buf250, buf251, primals_108, primals_109, buf254, 33554432, grid=grid(33554432), stream=stream0)
        del buf251
        del primals_109
        # Topologically Sorted Source Nodes: [y_22], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_110, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (4, 32, 64, 64, 64), (8388608, 262144, 4096, 64, 1))
        buf256 = buf255; del buf255  # reuse
        # Topologically Sorted Source Nodes: [y_22, y_23], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_7.run(buf256, primals_111, buf254, 33554432, grid=grid(33554432), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [y_24], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_112, stride=(1, 1, 1), padding=(1, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 4, 64, 64, 64), (1048576, 262144, 4096, 64, 1))
        buf258 = buf257; del buf257  # reuse
        # Topologically Sorted Source Nodes: [y_24], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_37.run(buf258, primals_113, 4194304, grid=grid(4194304), stream=stream0)
        del primals_113
    return (buf258, buf345, reinterpret_tensor(buf269, (4, 128), (256, 1), 0), reinterpret_tensor(buf269, (4, 128), (256, 1), 128), buf269, primals_1, primals_3, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, primals_18, primals_20, primals_22, primals_24, primals_26, primals_28, primals_30, primals_32, primals_34, primals_36, primals_38, primals_40, primals_42, primals_44, primals_46, primals_48, primals_50, primals_52, primals_54, primals_56, primals_58, primals_60, primals_62, primals_64, primals_66, primals_68, primals_70, primals_72, primals_74, primals_76, primals_78, primals_80, primals_82, primals_84, primals_86, primals_88, primals_90, primals_92, primals_94, primals_96, primals_98, primals_100, primals_102, primals_104, primals_106, primals_108, primals_110, primals_112, primals_114, primals_116, primals_122, primals_124, primals_126, primals_128, primals_130, primals_132, primals_134, primals_136, primals_138, primals_140, primals_142, primals_144, primals_146, primals_148, primals_150, primals_152, primals_154, buf1, reinterpret_tensor(buf5, (4, 8), (8, 1), 0), reinterpret_tensor(buf8, (4, 8), (8, 1), 0), buf9, buf11, reinterpret_tensor(buf15, (4, 8), (8, 1), 0), reinterpret_tensor(buf18, (4, 8), (8, 1), 0), buf19, buf21, buf23, reinterpret_tensor(buf27, (4, 8), (8, 1), 0), reinterpret_tensor(buf30, (4, 8), (8, 1), 0), buf31, buf33, reinterpret_tensor(buf37, (4, 8), (8, 1), 0), reinterpret_tensor(buf40, (4, 8), (8, 1), 0), buf41, buf43, reinterpret_tensor(buf47, (4, 8), (8, 1), 0), reinterpret_tensor(buf50, (4, 8), (8, 1), 0), buf51, buf53, reinterpret_tensor(buf57, (4, 8), (8, 1), 0), reinterpret_tensor(buf60, (4, 8), (8, 1), 0), buf61, buf63, buf65, reinterpret_tensor(buf69, (4, 8), (8, 1), 0), reinterpret_tensor(buf72, (4, 8), (8, 1), 0), buf73, buf75, reinterpret_tensor(buf79, (4, 8), (8, 1), 0), reinterpret_tensor(buf82, (4, 8), (8, 1), 0), buf83, buf85, reinterpret_tensor(buf89, (4, 8), (8, 1), 0), reinterpret_tensor(buf92, (4, 8), (8, 1), 0), buf93, buf95, reinterpret_tensor(buf99, (4, 8), (8, 1), 0), reinterpret_tensor(buf102, (4, 8), (8, 1), 0), buf103, buf105, buf107, reinterpret_tensor(buf111, (4, 8), (8, 1), 0), reinterpret_tensor(buf114, (4, 8), (8, 1), 0), buf115, buf117, reinterpret_tensor(buf121, (4, 8), (8, 1), 0), reinterpret_tensor(buf124, (4, 8), (8, 1), 0), buf125, buf127, reinterpret_tensor(buf131, (4, 8), (8, 1), 0), reinterpret_tensor(buf134, (4, 8), (8, 1), 0), buf135, buf137, reinterpret_tensor(buf141, (4, 8), (8, 1), 0), reinterpret_tensor(buf144, (4, 8), (8, 1), 0), buf145, buf147, reinterpret_tensor(buf151, (4, 8), (8, 1), 0), reinterpret_tensor(buf154, (4, 8), (8, 1), 0), buf155, buf157, reinterpret_tensor(buf161, (4, 8), (8, 1), 0), reinterpret_tensor(buf164, (4, 8), (8, 1), 0), buf165, buf167, reinterpret_tensor(buf171, (4, 8), (8, 1), 0), reinterpret_tensor(buf174, (4, 8), (8, 1), 0), buf175, buf177, reinterpret_tensor(buf181, (4, 8), (8, 1), 0), reinterpret_tensor(buf184, (4, 8), (8, 1), 0), buf185, buf187, buf189, buf190, reinterpret_tensor(buf194, (4, 8), (8, 1), 0), reinterpret_tensor(buf197, (4, 8), (8, 1), 0), buf198, buf200, reinterpret_tensor(buf204, (4, 8), (8, 1), 0), reinterpret_tensor(buf207, (4, 8), (8, 1), 0), buf208, buf210, buf212, buf213, reinterpret_tensor(buf217, (4, 8), (8, 1), 0), reinterpret_tensor(buf220, (4, 8), (8, 1), 0), buf221, buf223, reinterpret_tensor(buf227, (4, 8), (8, 1), 0), reinterpret_tensor(buf230, (4, 8), (8, 1), 0), buf231, buf233, buf235, buf236, reinterpret_tensor(buf240, (4, 8), (8, 1), 0), reinterpret_tensor(buf243, (4, 8), (8, 1), 0), buf244, buf246, reinterpret_tensor(buf250, (4, 8), (8, 1), 0), reinterpret_tensor(buf253, (4, 8), (8, 1), 0), buf254, buf256, reinterpret_tensor(buf262, (4, 8), (8, 1), 0), reinterpret_tensor(buf265, (4, 8), (8, 1), 0), buf266, reinterpret_tensor(buf268, (4, 1024), (1024, 1), 0), reinterpret_tensor(buf269, (4, 128), (256, 1), 128), buf271, buf272, buf274, buf276, buf277, buf279, reinterpret_tensor(buf283, (4, 8), (8, 1), 0), reinterpret_tensor(buf286, (4, 8), (8, 1), 0), buf287, buf289, reinterpret_tensor(buf293, (4, 8), (8, 1), 0), reinterpret_tensor(buf296, (4, 8), (8, 1), 0), buf297, buf299, buf301, reinterpret_tensor(buf305, (4, 8), (8, 1), 0), reinterpret_tensor(buf308, (4, 8), (8, 1), 0), buf309, buf311, reinterpret_tensor(buf315, (4, 8), (8, 1), 0), reinterpret_tensor(buf318, (4, 8), (8, 1), 0), buf319, buf321, buf323, reinterpret_tensor(buf327, (4, 8), (8, 1), 0), reinterpret_tensor(buf330, (4, 8), (8, 1), 0), buf331, buf333, reinterpret_tensor(buf337, (4, 8), (8, 1), 0), reinterpret_tensor(buf340, (4, 8), (8, 1), 0), buf341, buf343, primals_120, primals_118, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 2, 3, 3, 3), (54, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 2, 64, 64, 64), (524288, 262144, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((32, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((32, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((4, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((16, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, 16, 1, 1, 1), (16, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((64, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((64, 64, 3, 3, 3), (1728, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((32, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((32, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((32, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((2, 32, 3, 3, 3), (864, 27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
