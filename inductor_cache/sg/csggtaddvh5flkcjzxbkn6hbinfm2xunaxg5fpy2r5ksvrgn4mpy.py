# AOT ID: ['12_forward']
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


# kernel path: inductor_cache/qq/cqqetrenmnb4ytqbyb3cnazgq3phsxepqqrg5zb5fgzizxui6fcj.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%permute, %primals_2, None, [1], [0], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_0 = async_compile.triton('triton_poi_fused_convolution_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/y7/cy7dv7t2w3w2qx2aksar5nbzvum2diqhkaa7er46tes3gr4xr7wx.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.native_group_norm]
# Source node to ATen node mapping:
#   output => add, add_1, mul_1, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%view, [2, 3]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_1, %unsqueeze_3), kwargs = {})
#   %add_1 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %unsqueeze_1), kwargs = {})
triton_per_fused_native_group_norm_1 = async_compile.triton('triton_per_fused_native_group_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_group_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r3), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (r3), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 16, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 16.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.store(out_ptr2 + (r1 + 16*x0), tmp27, xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e7/ce766upql5syhv73ctoqfvr5dfpi4cnm46b72zr5exmtouhlrazj.py
# Topologically Sorted Source Nodes: [sub, view_1, d, idx], Original ATen: [aten.sub, aten.view, aten.linalg_vector_norm, aten.argmin]
# Source node to ATen node mapping:
#   d => pow_1, pow_2, sum_1
#   idx => argmin
#   sub => sub_1
#   view_1 => view_3
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_4, %unsqueeze_6), kwargs = {})
#   %view_3 : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%sub_1, [4, 4, 4, 1, -1]), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_3, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-1]), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
#   %argmin : [num_users=2] = call_function[target=torch.ops.aten.argmin.default](args = (%pow_2, 0), kwargs = {})
triton_poi_fused_argmin_linalg_vector_norm_sub_view_2 = async_compile.triton('triton_poi_fused_argmin_linalg_vector_norm_sub_view_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_argmin_linalg_vector_norm_sub_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_argmin_linalg_vector_norm_sub_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (4 + x0 + 16*x1), xmask)
    tmp6 = tl.load(in_ptr1 + (1))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (8 + x0 + 16*x1), xmask)
    tmp12 = tl.load(in_ptr1 + (2))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (12 + x0 + 16*x1), xmask)
    tmp18 = tl.load(in_ptr1 + (3))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp24 = tl.load(in_ptr1 + (4))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp28 = tl.load(in_ptr1 + (5))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK])
    tmp33 = tl.load(in_ptr1 + (6))
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK])
    tmp38 = tl.load(in_ptr1 + (7))
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK])
    tmp59 = tl.load(in_ptr1 + (8))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp63 = tl.load(in_ptr1 + (9))
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK])
    tmp68 = tl.load(in_ptr1 + (10))
    tmp69 = tl.broadcast_to(tmp68, [XBLOCK])
    tmp73 = tl.load(in_ptr1 + (11))
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK])
    tmp93 = tl.load(in_ptr1 + (12))
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK])
    tmp97 = tl.load(in_ptr1 + (13))
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK])
    tmp102 = tl.load(in_ptr1 + (14))
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK])
    tmp107 = tl.load(in_ptr1 + (15))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp3 = tmp0 - tmp2
    tmp4 = tmp3 * tmp3
    tmp8 = tmp5 - tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tmp4 + tmp9
    tmp14 = tmp11 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tmp10 + tmp15
    tmp20 = tmp17 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tmp16 + tmp21
    tmp23 = libdevice.sqrt(tmp22)
    tmp26 = tmp0 - tmp25
    tmp27 = tmp26 * tmp26
    tmp30 = tmp5 - tmp29
    tmp31 = tmp30 * tmp30
    tmp32 = tmp27 + tmp31
    tmp35 = tmp11 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tmp32 + tmp36
    tmp40 = tmp17 - tmp39
    tmp41 = tmp40 * tmp40
    tmp42 = tmp37 + tmp41
    tmp43 = libdevice.sqrt(tmp42)
    tmp44 = tmp23 < tmp43
    tmp45 = tmp23 == tmp43
    tmp46 = tmp23 != tmp23
    tmp47 = tmp43 != tmp43
    tmp48 = tmp46 > tmp47
    tmp49 = tmp44 | tmp48
    tmp50 = tmp46 & tmp47
    tmp51 = tmp45 | tmp50
    tmp52 = tl.full([1], 0, tl.int64)
    tmp53 = tl.full([1], 1, tl.int64)
    tmp54 = tmp52 < tmp53
    tmp55 = tmp51 & tmp54
    tmp56 = tmp49 | tmp55
    tmp57 = tl.where(tmp56, tmp23, tmp43)
    tmp58 = tl.where(tmp56, tmp52, tmp53)
    tmp61 = tmp0 - tmp60
    tmp62 = tmp61 * tmp61
    tmp65 = tmp5 - tmp64
    tmp66 = tmp65 * tmp65
    tmp67 = tmp62 + tmp66
    tmp70 = tmp11 - tmp69
    tmp71 = tmp70 * tmp70
    tmp72 = tmp67 + tmp71
    tmp75 = tmp17 - tmp74
    tmp76 = tmp75 * tmp75
    tmp77 = tmp72 + tmp76
    tmp78 = libdevice.sqrt(tmp77)
    tmp79 = tmp57 < tmp78
    tmp80 = tmp57 == tmp78
    tmp81 = tmp57 != tmp57
    tmp82 = tmp78 != tmp78
    tmp83 = tmp81 > tmp82
    tmp84 = tmp79 | tmp83
    tmp85 = tmp81 & tmp82
    tmp86 = tmp80 | tmp85
    tmp87 = tl.full([1], 2, tl.int64)
    tmp88 = tmp58 < tmp87
    tmp89 = tmp86 & tmp88
    tmp90 = tmp84 | tmp89
    tmp91 = tl.where(tmp90, tmp57, tmp78)
    tmp92 = tl.where(tmp90, tmp58, tmp87)
    tmp95 = tmp0 - tmp94
    tmp96 = tmp95 * tmp95
    tmp99 = tmp5 - tmp98
    tmp100 = tmp99 * tmp99
    tmp101 = tmp96 + tmp100
    tmp104 = tmp11 - tmp103
    tmp105 = tmp104 * tmp104
    tmp106 = tmp101 + tmp105
    tmp109 = tmp17 - tmp108
    tmp110 = tmp109 * tmp109
    tmp111 = tmp106 + tmp110
    tmp112 = libdevice.sqrt(tmp111)
    tmp113 = tmp91 < tmp112
    tmp114 = tmp91 == tmp112
    tmp115 = tmp91 != tmp91
    tmp116 = tmp112 != tmp112
    tmp117 = tmp115 > tmp116
    tmp118 = tmp113 | tmp117
    tmp119 = tmp115 & tmp116
    tmp120 = tmp114 | tmp119
    tmp121 = tl.full([1], 3, tl.int64)
    tmp122 = tmp92 < tmp121
    tmp123 = tmp120 & tmp122
    tmp124 = tmp118 | tmp123
    tmp125 = tl.where(tmp124, tmp91, tmp112)
    tmp126 = tl.where(tmp124, tmp92, tmp121)
    tl.store(out_ptr0 + (x2), tmp126, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gf/cgfmdiki2tz4o7pjajlafnxdevqfsvap3ehewu4b6y65ufbyf3u6.py
# Topologically Sorted Source Nodes: [latent_loss, commitment_loss, mul_1, add_2], Original ATen: [aten.mse_loss, aten.mul, aten.add]
# Source node to ATen node mapping:
#   add_2 => add_4
#   commitment_loss => mean_2, pow_4, sub_4
#   latent_loss => mean_1, pow_3, sub_3
#   mul_1 => mul_3
# Graph fragment:
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_2, %add_1), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_3, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_3,), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %permute_2), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_4, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%pow_4,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean_2, 0.25), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, %mul_3), kwargs = {})
triton_per_fused_add_mse_loss_mul_3 = async_compile.triton('triton_per_fused_add_mse_loss_mul_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': (4,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mse_loss_mul_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mse_loss_mul_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = (rindex % 4)
    r2 = rindex // 16
    r1 = ((rindex // 4) % 4)
    r3 = rindex
    tmp0 = tl.load(in_ptr0 + (r0 + 4*r2), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r3), None)
    tmp1 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (r1 + 4*tmp4), None, eviction_policy='evict_last')
    tmp8 = tmp6 - tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None]
    tmp13 = tmp7 - tmp6
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None]
    tmp18 = 64.0
    tmp19 = tmp12 / tmp18
    tmp20 = tmp17 / tmp18
    tmp21 = 0.25
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/3e/c3ezhzrg6cgzvunpoooyabn5pbvfpshquguxbqebqv3epessxbb5.py
# Topologically Sorted Source Nodes: [float_4, hard_probs], Original ATen: [aten._to_copy, aten.mean]
# Source node to ATen node mapping:
#   float_4 => convert_element_type
#   hard_probs => mean
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convert_element_type, [0]), kwargs = {})
triton_per_fused__to_copy_mean_4 = async_compile.triton('triton_per_fused__to_copy_mean_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_4(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp0 == tmp1
    tmp3 = tl.full([1, 1], 1, tl.int64)
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ma/cma7d7twcfalo4pob6l7vncfyvdxxrfpk7i3plpcazqzzrl467vw.py
# Topologically Sorted Source Nodes: [float_4, hard_probs, add_1, log, mul, sum_1, neg, exp, sum_2], Original ATen: [aten._to_copy, aten.mean, aten.add, aten.log, aten.mul, aten.sum, aten.neg, aten.exp]
# Source node to ATen node mapping:
#   add_1 => add_3
#   exp => exp
#   float_4 => convert_element_type
#   hard_probs => mean
#   log => log
#   mul => mul_2
#   neg => neg
#   sum_1 => sum_2
#   sum_2 => sum_3
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%convert_element_type, [0]), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-07), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mean, %log), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_2, [-1]), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_2,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%neg,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%exp,), kwargs = {})
triton_poi_fused__to_copy_add_exp_log_mean_mul_neg_sum_5 = async_compile.triton('triton_poi_fused__to_copy_add_exp_log_mean_mul_neg_sum_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_exp_log_mean_mul_neg_sum_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_exp_log_mean_mul_neg_sum_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp15 = tl.load(in_ptr0 + (2))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (3))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp2 = 16.0
    tmp3 = tmp1 / tmp2
    tmp4 = 1e-07
    tmp5 = tmp3 + tmp4
    tmp6 = tl_math.log(tmp5)
    tmp7 = tmp3 * tmp6
    tmp10 = tmp9 / tmp2
    tmp11 = tmp10 + tmp4
    tmp12 = tl_math.log(tmp11)
    tmp13 = tmp10 * tmp12
    tmp14 = tmp7 + tmp13
    tmp17 = tmp16 / tmp2
    tmp18 = tmp17 + tmp4
    tmp19 = tl_math.log(tmp18)
    tmp20 = tmp17 * tmp19
    tmp21 = tmp14 + tmp20
    tmp24 = tmp23 / tmp2
    tmp25 = tmp24 + tmp4
    tmp26 = tl_math.log(tmp25)
    tmp27 = tmp24 * tmp26
    tmp28 = tmp21 + tmp27
    tmp29 = -tmp28
    tmp30 = tl_math.exp(tmp29)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp30, None)
''', device_str='cuda')


# kernel path: inductor_cache/ei/ceivykqhp26v766bdqeydqpase5lvxhjrulswfv7fmmsdq4y6sex.py
# Topologically Sorted Source Nodes: [sub_1, x_1, latent_loss, commitment_loss], Original ATen: [aten.sub, aten.add, aten.mse_loss, aten.mse_loss_backward]
# Source node to ATen node mapping:
#   commitment_loss => sub_4
#   latent_loss => sub_3
#   sub_1 => sub_2
#   x_1 => add_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %add_1), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_2, %sub_2), kwargs = {})
#   %sub_3 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%permute_2, %add_1), kwargs = {})
#   %sub_4 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %permute_2), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, 0.03125), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, 0.03125), kwargs = {})
triton_poi_fused_add_mse_loss_mse_loss_backward_sub_6 = async_compile.triton('triton_poi_fused_add_mse_loss_mse_loss_backward_sub_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mse_loss_mse_loss_backward_sub_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mse_loss_mse_loss_backward_sub_6(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex // 4
    y0 = (yindex % 4)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y1), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK, YBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask & ymask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (y0 + 4*tmp4), xmask & ymask)
    tmp8 = tmp7 - tmp7
    tmp9 = tmp6 + tmp8
    tmp10 = tmp7 - tmp6
    tmp11 = 0.03125
    tmp12 = tmp10 * tmp11
    tmp13 = tmp6 - tmp7
    tmp14 = tmp13 * tmp11
    tl.store(out_ptr0 + (x2 + 4*y3), tmp9, xmask & ymask)
    tl.store(out_ptr1 + (x2 + 4*y3), tmp12, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 4*x2 + 16*y1), tmp14, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 1), (4, 1, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, 1, 4), (4, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(primals_1, buf0, 16, 4, grid=grid(16, 4), stream=stream0)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf1, (4, 4, 4), (16, 4, 1))
        buf2 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf5 = buf0; del buf0  # reuse
        buf6 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.native_group_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_group_norm_1.run(buf1, primals_3, primals_4, buf2, buf5, buf6, 4, 16, grid=grid(4), stream=stream0)
        del primals_4
        buf7 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.int64)
        # Topologically Sorted Source Nodes: [sub, view_1, d, idx], Original ATen: [aten.sub, aten.view, aten.linalg_vector_norm, aten.argmin]
        stream0 = get_raw_stream(0)
        triton_poi_fused_argmin_linalg_vector_norm_sub_view_2.run(buf5, primals_5, buf7, 16, grid=grid(16), stream=stream0)
        buf11 = empty_strided_cuda((), (), torch.float32)
        buf16 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [latent_loss, commitment_loss, mul_1, add_2], Original ATen: [aten.mse_loss, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mse_loss_mul_3.run(buf16, buf7, primals_5, buf5, 1, 64, grid=grid(1), stream=stream0)
        buf8 = empty_strided_cuda((1, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [float_4, hard_probs], Original ATen: [aten._to_copy, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_4.run(buf7, buf8, 4, 16, grid=grid(4), stream=stream0)
        buf15 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [float_4, hard_probs, add_1, log, mul, sum_1, neg, exp, sum_2], Original ATen: [aten._to_copy, aten.mean, aten.add, aten.log, aten.mul, aten.sum, aten.neg, aten.exp]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_exp_log_mean_mul_neg_sum_5.run(buf8, buf15, 1, grid=grid(1), stream=stream0)
        del buf8
        buf9 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf13 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        buf14 = empty_strided_cuda((4, 4, 4), (16, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [sub_1, x_1, latent_loss, commitment_loss], Original ATen: [aten.sub, aten.add, aten.mse_loss, aten.mse_loss_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mse_loss_mse_loss_backward_sub_6.run(buf7, primals_5, buf5, buf9, buf13, buf14, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_5
        buf10 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [sub_1, x_1, x_2], Original ATen: [aten.sub, aten.add, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_0.run(buf9, buf10, 16, 4, grid=grid(16, 4), stream=stream0)
        del buf9
    return (buf15, buf10, buf16, primals_2, primals_3, reinterpret_tensor(primals_1, (4, 4, 4), (16, 1, 4), 0), buf1, reinterpret_tensor(buf2, (4, 1), (1, 1), 0), reinterpret_tensor(buf6, (4, 1), (1, 1), 0), reinterpret_tensor(buf7, (4, 4), (4, 1), 0), buf13, buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 1), (4, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 1, 4), (4, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
