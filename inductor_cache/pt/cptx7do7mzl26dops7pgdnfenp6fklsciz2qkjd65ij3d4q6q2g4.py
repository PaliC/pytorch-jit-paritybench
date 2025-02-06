# AOT ID: ['7_forward']
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


# kernel path: inductor_cache/qw/cqwbgmaxvtbkhtmgppbsgq5bp3cyfva2phv4x6qm3a6sp6j6jjzv.py
# Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, input_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   input_1 => mul
#   mean => mean
#   pow_1 => pow_1
#   rsqrt => rsqrt
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-08), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, %rsqrt), kwargs = {})
triton_poi_fused_add_mean_mul_pow_rsqrt_0 = async_compile.triton('triton_poi_fused_add_mean_mul_pow_rsqrt_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_mul_pow_rsqrt_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_mul_pow_rsqrt_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-08
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp0 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fa/cfamou7gcgudbareniupeanvwfjlfhreswkqpz23rcqkotdz36mu.py
# Topologically Sorted Source Nodes: [mul_1, out], Original ATen: [aten.mul, aten.t]
# Source node to ATen node mapping:
#   mul_1 => mul_1
#   out => permute
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_2, 0.005), kwargs = {})
#   %permute : [num_users=4] = call_function[target=torch.ops.aten.permute.default](args = (%mul_1, [1, 0]), kwargs = {})
triton_poi_fused_mul_t_1 = async_compile.triton('triton_poi_fused_mul_t_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_t_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.005
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tt/cttdpv33nlq7zr2mnhyxnlkifjx4zlxgr3rorygcgxsgirzyluk4.py
# Topologically Sorted Source Nodes: [pow_2, mean_1, add_5, rsqrt_1, input_2], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_5 => add_5
#   input_2 => mul_17
#   mean_1 => mean_1
#   pow_2 => pow_2
#   rsqrt_1 => rsqrt_1
# Graph fragment:
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_1, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [1], True), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_1, 1e-08), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %mul_17 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_1, %rsqrt_1), kwargs = {})
triton_poi_fused_add_mean_mul_pow_rsqrt_2 = async_compile.triton('triton_poi_fused_add_mean_mul_pow_rsqrt_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_mul_pow_rsqrt_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_mul_pow_rsqrt_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (16 + x2), xmask)
    tmp1 = tl.load(in_ptr0 + (16 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (17 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (18 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (19 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-08
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp0 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vn/cvnu3k5zpdji46sw5b7s5ncl4iovi5iuyp5wrrclxodd6ulkviee.py
# Topologically Sorted Source Nodes: [pow_3, mean_2, add_10, rsqrt_2, input_3], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_10 => add_10
#   input_3 => mul_34
#   mean_2 => mean_2
#   pow_3 => pow_3
#   rsqrt_2 => rsqrt_2
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [1], True), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-08), kwargs = {})
#   %rsqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %mul_34 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %rsqrt_2), kwargs = {})
triton_poi_fused_add_mean_mul_pow_rsqrt_3 = async_compile.triton('triton_poi_fused_add_mean_mul_pow_rsqrt_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_mul_pow_rsqrt_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_mul_pow_rsqrt_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (32 + x2), xmask)
    tmp1 = tl.load(in_ptr0 + (32 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (33 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (34 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (35 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-08
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp0 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f4/cf44ra664o6wpoafohioumywnyqgxoz4fc72ld3hw5v6yvexguss.py
# Topologically Sorted Source Nodes: [pow_4, mean_3, add_15, rsqrt_3, input_4], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add_15 => add_15
#   input_4 => mul_51
#   mean_3 => mean_3
#   pow_4 => pow_4
#   rsqrt_3 => rsqrt_3
# Graph fragment:
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_3, 2), kwargs = {})
#   %mean_3 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [1], True), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_3, 1e-08), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_15,), kwargs = {})
#   %mul_51 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_3, %rsqrt_3), kwargs = {})
triton_poi_fused_add_mean_mul_pow_rsqrt_4 = async_compile.triton('triton_poi_fused_add_mean_mul_pow_rsqrt_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mean_mul_pow_rsqrt_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mean_mul_pow_rsqrt_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (48 + x2), xmask)
    tmp1 = tl.load(in_ptr0 + (48 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (49 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (50 + 4*x1), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (51 + 4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = 4.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-08
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.rsqrt(tmp15)
    tmp17 = tmp0 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rj/crj5oclm2olcigx3bn7tdlb5n5sjwkpfmjdgfvlchh7qawm33qfy.py
# Topologically Sorted Source Nodes: [mul_2, add_1, leaky_relu, out_1, add_6, leaky_relu_4, out_9, add_11, leaky_relu_8, out_17, add_16, leaky_relu_12, out_25], Original ATen: [aten.mul, aten.add, aten.leaky_relu]
# Source node to ATen node mapping:
#   add_1 => add_1
#   add_11 => add_11
#   add_16 => add_16
#   add_6 => add_6
#   leaky_relu => gt, mul_3, where
#   leaky_relu_12 => gt_12, mul_54, where_12
#   leaky_relu_4 => gt_4, mul_20, where_4
#   leaky_relu_8 => gt_8, mul_37, where_8
#   mul_2 => mul_2
#   out_1 => mul_4
#   out_17 => mul_38
#   out_25 => mul_55
#   out_9 => mul_21
# Graph fragment:
#   %mul_2 : [num_users=4] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_3, 0.01), kwargs = {})
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm, %mul_2), kwargs = {})
#   %gt : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_1, 0), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_1, 0.2), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %add_1, %mul_3), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where, 1.4142135623730951), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_4, %mul_2), kwargs = {})
#   %gt_4 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_6, 0), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_6, 0.2), kwargs = {})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_4, %add_6, %mul_20), kwargs = {})
#   %mul_21 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_4, 1.4142135623730951), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_8, %mul_2), kwargs = {})
#   %gt_8 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_11, 0), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_11, 0.2), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_8, %add_11, %mul_37), kwargs = {})
#   %mul_38 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_8, 1.4142135623730951), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_12, %mul_2), kwargs = {})
#   %gt_12 : [num_users=2] = call_function[target=torch.ops.aten.gt.Scalar](args = (%add_16, 0), kwargs = {})
#   %mul_54 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_16, 0.2), kwargs = {})
#   %where_12 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt_12, %add_16, %mul_54), kwargs = {})
#   %mul_55 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%where_12, 1.4142135623730951), kwargs = {})
triton_poi_fused_add_leaky_relu_mul_5 = async_compile.triton('triton_poi_fused_add_leaky_relu_mul_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_out_ptr2': '*fp32', 'in_out_ptr3': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'out_ptr2': '*i1', 'out_ptr3': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_leaky_relu_mul_5', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2', 'in_out_ptr3'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_leaky_relu_mul_5(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr1 + (x2), xmask)
    tmp18 = tl.load(in_out_ptr2 + (x2), xmask)
    tmp24 = tl.load(in_out_ptr3 + (x2), xmask)
    tmp2 = 0.01
    tmp3 = tmp1 * tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = 0.0
    tmp6 = tmp4 > tmp5
    tmp7 = 0.2
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = 1.4142135623730951
    tmp11 = tmp9 * tmp10
    tmp13 = tmp12 + tmp3
    tmp14 = tmp13 > tmp5
    tmp15 = tmp13 * tmp7
    tmp16 = tl.where(tmp14, tmp13, tmp15)
    tmp17 = tmp16 * tmp10
    tmp19 = tmp18 + tmp3
    tmp20 = tmp19 > tmp5
    tmp21 = tmp19 * tmp7
    tmp22 = tl.where(tmp20, tmp19, tmp21)
    tmp23 = tmp22 * tmp10
    tmp25 = tmp24 + tmp3
    tmp26 = tmp25 > tmp5
    tmp27 = tmp25 * tmp7
    tmp28 = tl.where(tmp26, tmp25, tmp27)
    tmp29 = tmp28 * tmp10
    tl.store(out_ptr0 + (x2), tmp6, xmask)
    tl.store(in_out_ptr0 + (x2), tmp11, xmask)
    tl.store(out_ptr1 + (x2), tmp14, xmask)
    tl.store(in_out_ptr1 + (x2), tmp17, xmask)
    tl.store(out_ptr2 + (x2), tmp20, xmask)
    tl.store(in_out_ptr2 + (x2), tmp23, xmask)
    tl.store(out_ptr3 + (x2), tmp26, xmask)
    tl.store(in_out_ptr3 + (x2), tmp29, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, 4), (4, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_1, mean, add, rsqrt, input_1], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_mul_pow_rsqrt_0.run(primals_1, buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, out], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_t_1.run(primals_2, buf1, 16, grid=grid(16), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, buf1, out=buf2)
        buf17 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_2, mean_1, add_5, rsqrt_1, input_2], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_mul_pow_rsqrt_2.run(primals_1, buf17, 16, grid=grid(16), stream=stream0)
        buf18 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf17, buf1, out=buf18)
        buf30 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_3, mean_2, add_10, rsqrt_2, input_3], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_mul_pow_rsqrt_3.run(primals_1, buf30, 16, grid=grid(16), stream=stream0)
        buf31 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.mm]
        extern_kernels.mm(buf30, buf1, out=buf31)
        buf43 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [pow_4, mean_3, add_15, rsqrt_3, input_4], Original ATen: [aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mean_mul_pow_rsqrt_4.run(primals_1, buf43, 16, grid=grid(16), stream=stream0)
        del primals_1
        buf44 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf43, buf1, out=buf44)
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf4 = buf2; del buf2  # reuse
        buf19 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf20 = buf18; del buf18  # reuse
        buf32 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf33 = buf31; del buf31  # reuse
        buf45 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf46 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [mul_2, add_1, leaky_relu, out_1, add_6, leaky_relu_4, out_9, add_11, leaky_relu_8, out_17, add_16, leaky_relu_12, out_25], Original ATen: [aten.mul, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_mul_5.run(buf4, buf20, buf33, buf46, primals_3, buf3, buf19, buf32, buf45, 16, grid=grid(16), stream=stream0)
        del primals_3
        buf5 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [mul_4, out_2], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_t_1.run(primals_4, buf5, 16, grid=grid(16), stream=stream0)
        del primals_4
        buf6 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, buf5, out=buf6)
        buf21 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten.mm]
        extern_kernels.mm(buf20, buf5, out=buf21)
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten.mm]
        extern_kernels.mm(buf33, buf5, out=buf34)
        buf47 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten.mm]
        extern_kernels.mm(buf46, buf5, out=buf47)
        buf7 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf8 = buf6; del buf6  # reuse
        buf22 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf23 = buf21; del buf21  # reuse
        buf35 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf36 = buf34; del buf34  # reuse
        buf48 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf49 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [mul_5, add_2, leaky_relu_1, out_3, add_7, leaky_relu_5, out_11, add_12, leaky_relu_9, out_19, add_17, leaky_relu_13, out_27], Original ATen: [aten.mul, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_mul_5.run(buf8, buf23, buf36, buf49, primals_5, buf7, buf22, buf35, buf48, 16, grid=grid(16), stream=stream0)
        del primals_5
        buf9 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mul_7, out_4], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_t_1.run(primals_6, buf9, 16, grid=grid(16), stream=stream0)
        del primals_6
        buf10 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_4], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, buf9, out=buf10)
        buf24 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf23, buf9, out=buf24)
        buf37 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf36, buf9, out=buf37)
        buf50 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.mm]
        extern_kernels.mm(buf49, buf9, out=buf50)
        buf11 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf12 = buf10; del buf10  # reuse
        buf25 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf26 = buf24; del buf24  # reuse
        buf38 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf39 = buf37; del buf37  # reuse
        buf51 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf52 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [mul_8, add_3, leaky_relu_2, out_5, add_8, leaky_relu_6, out_13, add_13, leaky_relu_10, out_21, add_18, leaky_relu_14, out_29], Original ATen: [aten.mul, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_mul_5.run(buf12, buf26, buf39, buf52, primals_7, buf11, buf25, buf38, buf51, 16, grid=grid(16), stream=stream0)
        del primals_7
        buf13 = empty_strided_cuda((4, 4), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [mul_10, out_6], Original ATen: [aten.mul, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_t_1.run(primals_8, buf13, 16, grid=grid(16), stream=stream0)
        del primals_8
        buf14 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten.mm]
        extern_kernels.mm(buf12, buf13, out=buf14)
        buf27 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.mm]
        extern_kernels.mm(buf26, buf13, out=buf27)
        buf40 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.mm]
        extern_kernels.mm(buf39, buf13, out=buf40)
        buf53 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.mm]
        extern_kernels.mm(buf52, buf13, out=buf53)
        buf15 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf16 = buf14; del buf14  # reuse
        buf28 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf29 = buf27; del buf27  # reuse
        buf41 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf42 = buf40; del buf40  # reuse
        buf54 = empty_strided_cuda((4, 4), (4, 1), torch.bool)
        buf55 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [mul_11, add_4, leaky_relu_3, out_7, add_9, leaky_relu_7, out_15, add_14, leaky_relu_11, out_23, add_19, leaky_relu_15, out_31], Original ATen: [aten.mul, aten.add, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_leaky_relu_mul_5.run(buf16, buf29, buf42, buf55, primals_9, buf15, buf28, buf41, buf54, 16, grid=grid(16), stream=stream0)
        del primals_9
    return (reinterpret_tensor(buf16, (4, 1, 4), (4, 4, 1), 0), buf16, buf29, buf42, buf55, buf0, buf3, buf4, buf7, buf8, buf11, buf12, buf15, buf17, buf19, buf20, buf22, buf23, buf25, buf26, buf28, buf30, buf32, buf33, buf35, buf36, buf38, buf39, buf41, buf43, buf45, buf46, buf48, buf49, buf51, buf52, buf54, reinterpret_tensor(buf13, (4, 4), (4, 1), 0), reinterpret_tensor(buf9, (4, 4), (4, 1), 0), reinterpret_tensor(buf5, (4, 4), (4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
