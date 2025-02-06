# AOT ID: ['54_inference']
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


# kernel path: inductor_cache/k7/ck7puy6axkf25sf5rgwfawhaahf4bbuhv6cuakgjvb7qul3k5cl3.py
# Topologically Sorted Source Nodes: [mu1, mu2, mul_3, mul_1, mul_2], Original ATen: [aten.convolution, aten.mul]
# Source node to ATen node mapping:
#   mu1 => convolution
#   mu2 => convolution_1
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg0_1, %arg1_1, None, [1, 1], [5, 5], [1, 1], False, [0, 0], 4), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%arg2_1, %arg1_1, None, [1, 1], [5, 5], [1, 1], False, [0, 0], 4), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg2_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, %arg0_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, %arg2_1), kwargs = {})
triton_poi_fused_convolution_mul_0 = async_compile.triton('triton_poi_fused_convolution_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask)
    tmp1 = tl.load(in_ptr1 + (x2 + 16*y3), xmask & ymask)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp0 * tmp0
    tmp4 = tmp1 * tmp1
    tl.store(out_ptr0 + (y0 + 4*x2 + 64*y1), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 4*x2 + 64*y1), tmp1, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 4*x2 + 64*y1), tmp2, xmask & ymask)
    tl.store(out_ptr3 + (y0 + 4*x2 + 64*y1), tmp3, xmask & ymask)
    tl.store(out_ptr4 + (y0 + 4*x2 + 64*y1), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/gi/cgi53xdhy5kunmoq36y5snft442sqwk4xx2nvien7qpvpgkdv5xa.py
# Topologically Sorted Source Nodes: [mu1_mu2, mul_4, add, sigma12, mul_5, add_1, mul_6, mu1_sq, mu2_sq, add_2, add_3, sigma1_sq, sigma2_sq, add_4, add_5, mul_7, ssim_map, mean], Original ATen: [aten.mul, aten.add, aten.sub, aten.pow, aten.div, aten.mean]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   mean => mean
#   mu1_mu2 => mul
#   mu1_sq => pow_1
#   mu2_sq => pow_2
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   sigma12 => sub_2
#   sigma1_sq => sub
#   sigma2_sq => sub_1
#   ssim_map => div
# Graph fragment:
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %convolution_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, 2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, 0.0001), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %mul), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, 2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, 0.0009), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add, %add_1), kwargs = {})
#   %pow_1 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution, 2), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_1, 2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_1, %pow_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, 0.0001), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %pow_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %pow_2), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub, %sub_1), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, 0.0009), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, %add_5), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%div, [1]), kwargs = {})
triton_poi_fused_add_div_mean_mul_pow_sub_1 = async_compile.triton('triton_poi_fused_add_div_mean_mul_pow_sub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_mul_pow_sub_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_mul_pow_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr4 + (4*x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp39 = tl.load(in_ptr3 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr4 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr2 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr3 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr4 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp71 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp76 = tl.load(in_ptr2 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr3 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr4 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 2.0
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0001
    tmp6 = tmp4 + tmp5
    tmp8 = tmp7 - tmp2
    tmp9 = tmp8 * tmp3
    tmp10 = 0.0009
    tmp11 = tmp9 + tmp10
    tmp12 = tmp6 * tmp11
    tmp13 = tmp0 * tmp0
    tmp14 = tmp1 * tmp1
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 + tmp5
    tmp18 = tmp17 - tmp13
    tmp20 = tmp19 - tmp14
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21 + tmp10
    tmp23 = tmp16 * tmp22
    tmp24 = tmp12 / tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27 * tmp3
    tmp29 = tmp28 + tmp5
    tmp31 = tmp30 - tmp27
    tmp32 = tmp31 * tmp3
    tmp33 = tmp32 + tmp10
    tmp34 = tmp29 * tmp33
    tmp35 = tmp25 * tmp25
    tmp36 = tmp26 * tmp26
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37 + tmp5
    tmp40 = tmp39 - tmp35
    tmp42 = tmp41 - tmp36
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43 + tmp10
    tmp45 = tmp38 * tmp44
    tmp46 = tmp34 / tmp45
    tmp47 = tmp24 + tmp46
    tmp50 = tmp48 * tmp49
    tmp51 = tmp50 * tmp3
    tmp52 = tmp51 + tmp5
    tmp54 = tmp53 - tmp50
    tmp55 = tmp54 * tmp3
    tmp56 = tmp55 + tmp10
    tmp57 = tmp52 * tmp56
    tmp58 = tmp48 * tmp48
    tmp59 = tmp49 * tmp49
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60 + tmp5
    tmp63 = tmp62 - tmp58
    tmp65 = tmp64 - tmp59
    tmp66 = tmp63 + tmp65
    tmp67 = tmp66 + tmp10
    tmp68 = tmp61 * tmp67
    tmp69 = tmp57 / tmp68
    tmp70 = tmp47 + tmp69
    tmp73 = tmp71 * tmp72
    tmp74 = tmp73 * tmp3
    tmp75 = tmp74 + tmp5
    tmp77 = tmp76 - tmp73
    tmp78 = tmp77 * tmp3
    tmp79 = tmp78 + tmp10
    tmp80 = tmp75 * tmp79
    tmp81 = tmp71 * tmp71
    tmp82 = tmp72 * tmp72
    tmp83 = tmp81 + tmp82
    tmp84 = tmp83 + tmp5
    tmp86 = tmp85 - tmp81
    tmp88 = tmp87 - tmp82
    tmp89 = tmp86 + tmp88
    tmp90 = tmp89 + tmp10
    tmp91 = tmp84 * tmp90
    tmp92 = tmp80 / tmp91
    tmp93 = tmp70 + tmp92
    tmp94 = 4.0
    tmp95 = tmp93 / tmp94
    tl.store(out_ptr0 + (x0), tmp95, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wt/cwtzl74iqw5os6ymda6cyr46c56nrl2p6ldylduwpupjtupkvoyy.py
# Topologically Sorted Source Nodes: [mean_1, batch_values], Original ATen: [aten.mean]
# Source node to ATen node mapping:
#   batch_values => mean_2
#   mean_1 => mean_1
# Graph fragment:
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%mean, [1]), kwargs = {})
#   %mean_2 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mean_1, [1]), kwargs = {})
triton_poi_fused_mean_2 = async_compile.triton('triton_poi_fused_mean_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mean_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mean_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (4 + 16*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (8 + 16*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (12 + 16*x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (1 + 16*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (5 + 16*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (9 + 16*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (13 + 16*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (2 + 16*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (6 + 16*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr0 + (10 + 16*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr0 + (14 + 16*x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr0 + (3 + 16*x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (7 + 16*x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr0 + (11 + 16*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (15 + 16*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp15 / tmp7
    tmp17 = tmp8 + tmp16
    tmp20 = tmp18 + tmp19
    tmp22 = tmp20 + tmp21
    tmp24 = tmp22 + tmp23
    tmp25 = tmp24 / tmp7
    tmp26 = tmp17 + tmp25
    tmp29 = tmp27 + tmp28
    tmp31 = tmp29 + tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33 / tmp7
    tmp35 = tmp26 + tmp34
    tmp36 = tmp35 / tmp7
    tl.store(out_ptr0 + (x0), tmp36, xmask)
''', device_str='cuda')


cpp_fused__to_copy_cat_3 = async_compile.cpp_pybinding(['const double*', 'const float*', 'double*', 'double*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(const double* in_ptr0,
                       const float* in_ptr1,
                       double* out_ptr0,
                       double* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(8L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr0[static_cast<int64_t>(x0)];
                    out_ptr0[static_cast<int64_t>(x0)] = tmp0;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(4L); x0+=static_cast<int64_t>(1L))
        {
            {
                {
                    auto tmp0 = in_ptr1[static_cast<int64_t>(x0)];
                    auto tmp1 = c10::convert<double>(tmp0);
                    out_ptr1[static_cast<int64_t>(x0)] = tmp1;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 1, 11, 11), (121, 121, 11, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg3_1, (8, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf4 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mu1, mu2, mul_3, mul_1, mul_2], Original ATen: [aten.convolution, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_0.run(arg0_1, arg2_1, buf0, buf2, buf4, buf6, buf8, 16, 16, grid=grid(16, 16), stream=stream0)
        del arg0_1
        del arg2_1
        # Topologically Sorted Source Nodes: [mu1], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf1, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf0
        # Topologically Sorted Source Nodes: [mu2], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf3, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf2
        # Topologically Sorted Source Nodes: [mul_3, conv2d_4], Original ATen: [aten.mul, aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf5, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf4
        # Topologically Sorted Source Nodes: [mul_1, conv2d_2], Original ATen: [aten.mul, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf7, (4, 4, 4, 4), (64, 1, 16, 4))
        del buf6
        # Topologically Sorted Source Nodes: [mul_2, conv2d_3], Original ATen: [aten.mul, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg1_1, stride=(1, 1), padding=(5, 5), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=4, bias=None)
        assert_size_stride(buf9, (4, 4, 4, 4), (64, 1, 16, 4))
        del arg1_1
        del buf8
        buf10 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mu1_mu2, mul_4, add, sigma12, mul_5, add_1, mul_6, mu1_sq, mu2_sq, add_2, add_3, sigma1_sq, sigma2_sq, add_4, add_5, mul_7, ssim_map, mean], Original ATen: [aten.mul, aten.add, aten.sub, aten.pow, aten.div, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_mul_pow_sub_1.run(buf1, buf3, buf5, buf7, buf9, buf10, 64, grid=grid(64), stream=stream0)
        del buf1
        del buf3
        del buf5
        del buf7
        del buf9
        buf11 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mean_1, batch_values], Original ATen: [aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mean_2.run(buf10, buf11, 4, grid=grid(4), stream=stream0)
        del buf10
    buf12 = empty_strided_cpu((4, ), (1, ), torch.float32)
    buf12.copy_(buf11, False)
    buf15 = empty_strided_cpu((12, ), (1, ), torch.float64)
    buf13 = reinterpret_tensor(buf15, (8, ), (1, ), 0)  # alias
    buf14 = reinterpret_tensor(buf15, (4, ), (1, ), 8)  # alias
    cpp_fused__to_copy_cat_3(arg3_1, buf12, buf13, buf14)
    del arg3_1
    return (buf11, buf15, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 1, 11, 11), (121, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8, ), (1, ), device='cpu', dtype=torch.float64)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
