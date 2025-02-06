# AOT ID: ['14_forward']
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


# kernel path: inductor_cache/j5/cj5dkvqq4imuvn4fkbeisd23jenhpf6geao5rr5hrlpikrq736yj.py
# Topologically Sorted Source Nodes: [wu], Original ATen: [aten.dot]
# Source node to ATen node mapping:
#   wu => mul, sum_1
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute, %primals_2), kwargs = {})
#   %sum_1 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
triton_poi_fused_dot_0 = async_compile.triton('triton_poi_fused_dot_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': (3,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_dot_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_dot_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp5 = tl.load(in_ptr0 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr1 + (1))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (2))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp13 = tl.load(in_ptr1 + (2))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK])
    tmp17 = tl.load(in_ptr0 + (3))
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (3))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp4 = tmp1 * tmp3
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 + tmp9
    tmp15 = tmp12 * tmp14
    tmp16 = tmp10 + tmp15
    tmp21 = tmp18 * tmp20
    tmp22 = tmp16 + tmp21
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/3h/c3hnrtctqt5zd4mn2ywwei7pis5kxl2rfhlke54bfctmciy5srmi.py
# Topologically Sorted Source Nodes: [exp, add, log, add_1, sub, norm, pow_1, add_2, truediv, mul, u], Original ATen: [aten.exp, aten.add, aten.log, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div, aten.mul]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   exp => exp
#   log => log
#   mul => mul_1
#   norm => pow_1, pow_2, sum_2
#   pow_1 => pow_3
#   sub => sub
#   truediv => div
#   u => add_3
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sum_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log, -1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %sum_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, None), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%pow_2, 2), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_3, 1e-15), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %add_2), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %div), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %mul_1), kwargs = {})
triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1 = async_compile.triton('triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp10 = tl.load(in_ptr2 + (x0), xmask)
    tmp11 = tl.load(in_ptr2 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp14 = tl.load(in_ptr2 + (1))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp18 = tl.load(in_ptr2 + (2))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp22 = tl.load(in_ptr2 + (3))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp3 = tl_math.exp(tmp2)
    tmp4 = 1.0
    tmp5 = tmp3 + tmp4
    tmp6 = tl_math.log(tmp5)
    tmp7 = -1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8 - tmp2
    tmp13 = tmp12 * tmp12
    tmp16 = tmp15 * tmp15
    tmp17 = tmp13 + tmp16
    tmp20 = tmp19 * tmp19
    tmp21 = tmp17 + tmp20
    tmp24 = tmp23 * tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = libdevice.sqrt(tmp25)
    tmp27 = tmp26 * tmp26
    tmp28 = 1e-15
    tmp29 = tmp27 + tmp28
    tmp30 = tmp10 / tmp29
    tmp31 = tmp9 * tmp30
    tmp32 = tmp0 + tmp31
    tl.store(out_ptr0 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/2w/c2wuivhwvoojo3flzkcugstynvpsefzkbckiq3a33zokearhi2iu.py
# Topologically Sorted Source Nodes: [hidden_units], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   hidden_units => add_4
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_4), kwargs = {})
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (16 + x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp5 = tl.load(in_ptr1 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (32 + x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (48 + x0 + 4*x1 + 64*x2 + ((x2 + 4*x1) // 16)), xmask)
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp19 = tl.load(in_ptr2 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp3 = tmp0 * tmp2
    tmp7 = tmp4 * tmp6
    tmp8 = tmp3 + tmp7
    tmp12 = tmp9 * tmp11
    tmp13 = tmp8 + tmp12
    tmp17 = tmp14 * tmp16
    tmp18 = tmp13 + tmp17
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr0 + (x3), tmp21, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yy/cyyjcywkv3vg6k5zavzus2iy6eerui7e4njanq76pwb72cnwqjry.py
# Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.mv]
# Source node to ATen node mapping:
#   matmul_4 => mul_8, sum_7
# Graph fragment:
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_4, %permute_6), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [1]), kwargs = {})
triton_poi_fused_mv_3 = async_compile.triton('triton_poi_fused_mv_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mv_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 13, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mv_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0 // 16), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp6 = tl.load(in_ptr3 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (16 + 4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (4 + 16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp14 = tl.load(in_ptr3 + (1))
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp18 = tl.load(in_ptr0 + (32 + 4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (8 + 16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp23 = tl.load(in_ptr3 + (2))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp27 = tl.load(in_ptr0 + (48 + 4*(((x0 // 4) % 4)) + 64*((x0 % 4)) + (x0 // 16)), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (12 + 16*(((x0 // 4) % 4)) + ((x0 % 4))), xmask)
    tmp32 = tl.load(in_ptr3 + (3))
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp8 = tmp5 * tmp7
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp1 * tmp11
    tmp13 = tmp9 + tmp12
    tmp16 = tmp13 * tmp15
    tmp17 = tmp8 + tmp16
    tmp20 = libdevice.tanh(tmp19)
    tmp21 = tmp1 * tmp20
    tmp22 = tmp18 + tmp21
    tmp25 = tmp22 * tmp24
    tmp26 = tmp17 + tmp25
    tmp29 = libdevice.tanh(tmp28)
    tmp30 = tmp1 * tmp29
    tmp31 = tmp27 + tmp30
    tmp34 = tmp31 * tmp33
    tmp35 = tmp26 + tmp34
    tl.store(out_ptr0 + (x0), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/h4/ch4jtceqwsfonq3dj4hjcdqgy7xi32pfk3rakfrffrpxxiblsed4.py
# Topologically Sorted Source Nodes: [mul_1, x, mul_4, x_1], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_1 => mul_3
#   mul_4 => mul_9
#   x => add_5
#   x_1 => add_14
# Graph fragment:
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze, %unsqueeze_1), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %mul_3), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_4, %unsqueeze_5), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %mul_9), kwargs = {})
triton_poi_fused_add_mul_4 = async_compile.triton('triton_poi_fused_add_mul_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x5 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x3 + 4*x2 + 16*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x5), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr5 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp6 * tmp11
    tmp13 = tmp5 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d2/cd2swwbn3xqqb4e2j4m6r4b6ke5j6xhu5rx7pcfn3if6qn6uvdia.py
# Topologically Sorted Source Nodes: [matmul_2, add_6, abs_1, add_7, log_det, log_jacobians, matmul_5, add_15, abs_2, add_16, log_det_1, log_jacobians_1, matmul_8, add_23, abs_3, add_24, log_det_2, log_jacobians_2, matmul_11, add_31, abs_4, add_32, log_det_3, log_jacobians_3], Original ATen: [aten.mv, aten.add, aten.abs, aten.log]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   abs_2 => abs_2
#   abs_3 => abs_3
#   abs_4 => abs_4
#   add_15 => add_15
#   add_16 => add_16
#   add_23 => add_24
#   add_24 => add_25
#   add_31 => add_33
#   add_32 => add_34
#   add_6 => add_6
#   add_7 => add_7
#   log_det => log_1
#   log_det_1 => log_3
#   log_det_2 => log_5
#   log_det_3 => log_7
#   log_jacobians => add_8
#   log_jacobians_1 => add_17
#   log_jacobians_2 => add_26
#   log_jacobians_3 => add_35
#   matmul_11 => mul_23, sum_16
#   matmul_2 => mul_5, sum_4
#   matmul_5 => mul_11, sum_8
#   matmul_8 => mul_17, sum_12
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_2, %permute_4), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [1]), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, 1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_6,), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_1, 1e-15), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_7,), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%log_1, 0), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_6, %permute_10), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_11, [1]), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_7, 1), kwargs = {})
#   %abs_2 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_15,), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_2, 1e-15), kwargs = {})
#   %log_3 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_16,), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %log_3), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_10, %permute_16), kwargs = {})
#   %sum_12 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_17, [1]), kwargs = {})
#   %add_24 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, 1), kwargs = {})
#   %abs_3 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_24,), kwargs = {})
#   %add_25 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_3, 1e-15), kwargs = {})
#   %log_5 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_25,), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_17, %log_5), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_14, %permute_22), kwargs = {})
#   %sum_16 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_23, [1]), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_15, 1), kwargs = {})
#   %abs_4 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_33,), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%abs_4, 1e-15), kwargs = {})
#   %log_7 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_34,), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_26, %log_7), kwargs = {})
triton_poi_fused_abs_add_log_mv_5 = async_compile.triton('triton_poi_fused_abs_add_log_mv_5', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_log_mv_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 50, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_log_mv_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp8 = tl.load(in_ptr2 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (4 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (1))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp18 = tl.load(in_ptr2 + (1))
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + (8 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr1 + (2))
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK])
    tmp29 = tl.load(in_ptr2 + (2))
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK])
    tmp33 = tl.load(in_ptr0 + (12 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr1 + (3))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp40 = tl.load(in_ptr2 + (3))
    tmp41 = tl.broadcast_to(tmp40, [XBLOCK])
    tmp44 = tl.load(in_ptr3 + (16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp45 = tl.load(in_ptr4 + (0))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp51 = tl.load(in_ptr5 + (0))
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK])
    tmp54 = tl.load(in_ptr6 + (0))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp57 = tl.load(in_ptr3 + (4 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp62 = tl.load(in_ptr5 + (1))
    tmp63 = tl.broadcast_to(tmp62, [XBLOCK])
    tmp65 = tl.load(in_ptr6 + (1))
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK])
    tmp69 = tl.load(in_ptr3 + (8 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp74 = tl.load(in_ptr5 + (2))
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK])
    tmp77 = tl.load(in_ptr6 + (2))
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK])
    tmp81 = tl.load(in_ptr3 + (12 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp86 = tl.load(in_ptr5 + (3))
    tmp87 = tl.broadcast_to(tmp86, [XBLOCK])
    tmp89 = tl.load(in_ptr6 + (3))
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK])
    tmp93 = tl.load(in_ptr7 + (16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr8 + (0))
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK])
    tmp100 = tl.load(in_ptr9 + (0))
    tmp101 = tl.broadcast_to(tmp100, [XBLOCK])
    tmp103 = tl.load(in_ptr7 + (4 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr8 + (1))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp110 = tl.load(in_ptr9 + (1))
    tmp111 = tl.broadcast_to(tmp110, [XBLOCK])
    tmp114 = tl.load(in_ptr7 + (8 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp118 = tl.load(in_ptr8 + (2))
    tmp119 = tl.broadcast_to(tmp118, [XBLOCK])
    tmp121 = tl.load(in_ptr9 + (2))
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK])
    tmp125 = tl.load(in_ptr7 + (12 + 16*((x0 % 4)) + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp129 = tl.load(in_ptr8 + (3))
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK])
    tmp132 = tl.load(in_ptr9 + (3))
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK])
    tmp136 = tl.load(in_ptr10 + (16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp137 = tl.load(in_ptr11 + (0))
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK])
    tmp143 = tl.load(in_ptr12 + (0))
    tmp144 = tl.broadcast_to(tmp143, [XBLOCK])
    tmp146 = tl.load(in_ptr13 + (0))
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK])
    tmp149 = tl.load(in_ptr10 + (4 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp154 = tl.load(in_ptr12 + (1))
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK])
    tmp157 = tl.load(in_ptr13 + (1))
    tmp158 = tl.broadcast_to(tmp157, [XBLOCK])
    tmp161 = tl.load(in_ptr10 + (8 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp166 = tl.load(in_ptr12 + (2))
    tmp167 = tl.broadcast_to(tmp166, [XBLOCK])
    tmp169 = tl.load(in_ptr13 + (2))
    tmp170 = tl.broadcast_to(tmp169, [XBLOCK])
    tmp173 = tl.load(in_ptr10 + (12 + 16*(x0 // 4) + ((x0 % 4))), xmask)
    tmp178 = tl.load(in_ptr12 + (3))
    tmp179 = tl.broadcast_to(tmp178, [XBLOCK])
    tmp181 = tl.load(in_ptr13 + (3))
    tmp182 = tl.broadcast_to(tmp181, [XBLOCK])
    tmp1 = libdevice.tanh(tmp0)
    tmp2 = tmp1 * tmp1
    tmp3 = 1.0
    tmp4 = tmp3 - tmp2
    tmp7 = tmp4 * tmp6
    tmp10 = tmp7 * tmp9
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = tmp12 * tmp12
    tmp14 = tmp3 - tmp13
    tmp17 = tmp14 * tmp16
    tmp20 = tmp17 * tmp19
    tmp21 = tmp10 + tmp20
    tmp23 = libdevice.tanh(tmp22)
    tmp24 = tmp23 * tmp23
    tmp25 = tmp3 - tmp24
    tmp28 = tmp25 * tmp27
    tmp31 = tmp28 * tmp30
    tmp32 = tmp21 + tmp31
    tmp34 = libdevice.tanh(tmp33)
    tmp35 = tmp34 * tmp34
    tmp36 = tmp3 - tmp35
    tmp39 = tmp36 * tmp38
    tmp42 = tmp39 * tmp41
    tmp43 = tmp32 + tmp42
    tmp47 = tmp44 + tmp46
    tmp48 = libdevice.tanh(tmp47)
    tmp49 = tmp48 * tmp48
    tmp50 = tmp3 - tmp49
    tmp53 = tmp50 * tmp52
    tmp56 = tmp53 * tmp55
    tmp58 = tmp57 + tmp46
    tmp59 = libdevice.tanh(tmp58)
    tmp60 = tmp59 * tmp59
    tmp61 = tmp3 - tmp60
    tmp64 = tmp61 * tmp63
    tmp67 = tmp64 * tmp66
    tmp68 = tmp56 + tmp67
    tmp70 = tmp69 + tmp46
    tmp71 = libdevice.tanh(tmp70)
    tmp72 = tmp71 * tmp71
    tmp73 = tmp3 - tmp72
    tmp76 = tmp73 * tmp75
    tmp79 = tmp76 * tmp78
    tmp80 = tmp68 + tmp79
    tmp82 = tmp81 + tmp46
    tmp83 = libdevice.tanh(tmp82)
    tmp84 = tmp83 * tmp83
    tmp85 = tmp3 - tmp84
    tmp88 = tmp85 * tmp87
    tmp91 = tmp88 * tmp90
    tmp92 = tmp80 + tmp91
    tmp94 = libdevice.tanh(tmp93)
    tmp95 = tmp94 * tmp94
    tmp96 = tmp3 - tmp95
    tmp99 = tmp96 * tmp98
    tmp102 = tmp99 * tmp101
    tmp104 = libdevice.tanh(tmp103)
    tmp105 = tmp104 * tmp104
    tmp106 = tmp3 - tmp105
    tmp109 = tmp106 * tmp108
    tmp112 = tmp109 * tmp111
    tmp113 = tmp102 + tmp112
    tmp115 = libdevice.tanh(tmp114)
    tmp116 = tmp115 * tmp115
    tmp117 = tmp3 - tmp116
    tmp120 = tmp117 * tmp119
    tmp123 = tmp120 * tmp122
    tmp124 = tmp113 + tmp123
    tmp126 = libdevice.tanh(tmp125)
    tmp127 = tmp126 * tmp126
    tmp128 = tmp3 - tmp127
    tmp131 = tmp128 * tmp130
    tmp134 = tmp131 * tmp133
    tmp135 = tmp124 + tmp134
    tmp139 = tmp136 + tmp138
    tmp140 = libdevice.tanh(tmp139)
    tmp141 = tmp140 * tmp140
    tmp142 = tmp3 - tmp141
    tmp145 = tmp142 * tmp144
    tmp148 = tmp145 * tmp147
    tmp150 = tmp149 + tmp138
    tmp151 = libdevice.tanh(tmp150)
    tmp152 = tmp151 * tmp151
    tmp153 = tmp3 - tmp152
    tmp156 = tmp153 * tmp155
    tmp159 = tmp156 * tmp158
    tmp160 = tmp148 + tmp159
    tmp162 = tmp161 + tmp138
    tmp163 = libdevice.tanh(tmp162)
    tmp164 = tmp163 * tmp163
    tmp165 = tmp3 - tmp164
    tmp168 = tmp165 * tmp167
    tmp171 = tmp168 * tmp170
    tmp172 = tmp160 + tmp171
    tmp174 = tmp173 + tmp138
    tmp175 = libdevice.tanh(tmp174)
    tmp176 = tmp175 * tmp175
    tmp177 = tmp3 - tmp176
    tmp180 = tmp177 * tmp179
    tmp183 = tmp180 * tmp182
    tmp184 = tmp172 + tmp183
    tmp185 = tmp43 + tmp3
    tmp186 = tl_math.abs(tmp185)
    tmp187 = 1e-15
    tmp188 = tmp186 + tmp187
    tmp189 = tl_math.log(tmp188)
    tmp190 = 0.0
    tmp191 = tmp189 + tmp190
    tmp192 = tmp92 + tmp3
    tmp193 = tl_math.abs(tmp192)
    tmp194 = tmp193 + tmp187
    tmp195 = tl_math.log(tmp194)
    tmp196 = tmp191 + tmp195
    tmp197 = tmp135 + tmp3
    tmp198 = tl_math.abs(tmp197)
    tmp199 = tmp198 + tmp187
    tmp200 = tl_math.log(tmp199)
    tmp201 = tmp196 + tmp200
    tmp202 = tmp184 + tmp3
    tmp203 = tl_math.abs(tmp202)
    tmp204 = tmp203 + tmp187
    tmp205 = tl_math.log(tmp204)
    tmp206 = tmp201 + tmp205
    tl.store(in_out_ptr0 + (x0), tmp206, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6z/c6zonm33frruc2py2ndj7z3yrutmpj7btjiz7wqhphv65lhn2s6k.py
# Topologically Sorted Source Nodes: [mul_7, x_2, mul_10, x_3], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_10 => mul_21
#   mul_7 => mul_15
#   x_2 => add_23
#   x_3 => add_32
# Graph fragment:
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_8, %unsqueeze_9), kwargs = {})
#   %add_23 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %mul_15), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_12, %unsqueeze_13), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_23, %mul_21), kwargs = {})
triton_poi_fused_add_mul_6 = async_compile.triton('triton_poi_fused_add_mul_6', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x5 = xindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x3 + 4*x2 + 16*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x5), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = libdevice.tanh(tmp10)
    tmp12 = tmp6 * tmp11
    tmp13 = tmp5 + tmp12
    tl.store(in_out_ptr0 + (x4), tmp13, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13 = args
    args.clear()
    assert_size_stride(primals_1, (4, ), (1, ))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (1, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (1, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (1, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [wu], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_0.run(primals_1, primals_2, buf0, 1, grid=grid(1), stream=stream0)
        buf1 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [exp, add, log, add_1, sub, norm, pow_1, add_2, truediv, mul, u], Original ATen: [aten.exp, aten.add, aten.log, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1.run(primals_2, buf0, primals_1, buf1, 4, grid=grid(4), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 4), (1, 4, 16), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_units], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(primals_3, primals_1, primals_4, buf2, 64, grid=grid(64), stream=stream0)
        buf9 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [wu_2], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_0.run(primals_8, primals_9, buf9, 1, grid=grid(1), stream=stream0)
        buf10 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [exp_2, add_17, log_4, add_18, sub_4, norm_2, pow_5, add_19, truediv_2, mul_6, u_2], Original ATen: [aten.exp, aten.add, aten.log, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1.run(primals_9, buf9, primals_8, buf10, 4, grid=grid(4), stream=stream0)
        buf4 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [wu_1], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_0.run(primals_5, primals_6, buf4, 1, grid=grid(1), stream=stream0)
        buf5 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [exp_1, add_9, log_2, add_10, sub_2, norm_1, pow_3, add_11, truediv_1, mul_3, u_1], Original ATen: [aten.exp, aten.add, aten.log, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1.run(primals_6, buf4, primals_5, buf5, 4, grid=grid(4), stream=stream0)
        buf6 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mv_3.run(primals_3, buf1, buf2, primals_5, buf6, 64, grid=grid(64), stream=stream0)
        buf7 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, x, mul_4, x_1], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_4.run(primals_3, buf1, buf2, buf5, buf6, primals_7, buf7, 256, grid=grid(256), stream=stream0)
        buf11 = empty_strided_cuda((4, 4, 4), (1, 4, 16), torch.float32)
        # Topologically Sorted Source Nodes: [hidden_units_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf7, primals_8, primals_10, buf11, 64, grid=grid(64), stream=stream0)
        buf13 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [wu_3], Original ATen: [aten.dot]
        stream0 = get_raw_stream(0)
        triton_poi_fused_dot_0.run(primals_11, primals_12, buf13, 1, grid=grid(1), stream=stream0)
        buf14 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [exp_3, add_25, log_6, add_26, sub_6, norm_3, pow_7, add_27, truediv_3, mul_9, u_3], Original ATen: [aten.exp, aten.add, aten.log, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_linalg_vector_norm_log_mul_pow_sub_1.run(primals_12, buf13, primals_11, buf14, 4, grid=grid(4), stream=stream0)
        del buf13
        buf15 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.mv]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mv_3.run(buf7, buf10, buf11, primals_11, buf15, 64, grid=grid(64), stream=stream0)
        buf3 = empty_strided_cuda((16, ), (1, ), torch.float32)
        buf18 = reinterpret_tensor(buf3, (1, 4, 4), (16, 4, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [matmul_2, add_6, abs_1, add_7, log_det, log_jacobians, matmul_5, add_15, abs_2, add_16, log_det_1, log_jacobians_1, matmul_8, add_23, abs_3, add_24, log_det_2, log_jacobians_2, matmul_11, add_31, abs_4, add_32, log_det_3, log_jacobians_3], Original ATen: [aten.mv, aten.add, aten.abs, aten.log]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_log_mv_5.run(buf18, buf2, primals_1, buf1, buf6, primals_7, primals_5, buf5, buf11, primals_8, buf10, buf15, primals_13, primals_11, buf14, 16, grid=grid(16), stream=stream0)
        del buf1
        del buf2
        del buf5
        del buf6
        buf16 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [mul_7, x_2, mul_10, x_3], Original ATen: [aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_6.run(buf16, buf10, buf11, buf14, buf15, primals_13, 256, grid=grid(256), stream=stream0)
        del buf10
        del buf11
        del buf14
        del buf15
    return (buf16, buf18, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
