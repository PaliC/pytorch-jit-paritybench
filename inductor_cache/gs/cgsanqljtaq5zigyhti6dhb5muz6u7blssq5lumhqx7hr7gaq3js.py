# AOT ID: ['5_forward']
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


# kernel path: inductor_cache/3u/c3uo2sgxywstvxbqkp2i2likjwhsqvyp4izp4h7x7xxhniz5z3ik.py
# Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.elu]
# Source node to ATen node mapping:
#   input_2 => expm1, gt, mul, mul_2, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%addmm, 0), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%addmm, 1.0), kwargs = {})
#   %expm1 : [num_users=1] = call_function[target=torch.ops.aten.expm1.default](args = (%mul,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%expm1, 1.0), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %mul, %mul_2), kwargs = {})
triton_poi_fused_elu_0 = async_compile.triton('triton_poi_fused_elu_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_elu_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_elu_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/px/cpxkffbwxci4lcqym7g4ai46doxeewpjhrjrd64jzrkpj3xkmfr7.py
# Topologically Sorted Source Nodes: [z1_norm], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   z1_norm => pow_1, pow_2, sum_1
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%addmm_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [-1], True), kwargs = {})
#   %pow_2 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_1, 0.5), kwargs = {})
triton_poi_fused_linalg_vector_norm_1 = async_compile.triton('triton_poi_fused_linalg_vector_norm_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_linalg_vector_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_linalg_vector_norm_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2zd3wiu62taqkpf7xcaq6imr7slskoczbfbf5nqyqkuo3rw4do.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view, 1e-08), kwargs = {})
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp3 = 0.25
    tmp4 = tmp2 * tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp3
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 / tmp13
    tmp15 = tmp14 * tmp3
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp11 + tmp16
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20 * tmp3
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp17 + tmp22
    tmp24 = 1e-08
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/u3/cu3dzxrv6iwdpacwxp6eh5i6zfiur47qdbdzhjilndckhixbegbh.py
# Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_1 => add_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, 1e-08), kwargs = {})
triton_poi_fused_add_3 = async_compile.triton('triton_poi_fused_add_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp6 = tl.load(in_ptr0 + (4 + x0), xmask)
    tmp7 = tl.load(in_ptr1 + (4 + x0), xmask)
    tmp12 = tl.load(in_ptr0 + (8 + x0), xmask)
    tmp13 = tl.load(in_ptr1 + (8 + x0), xmask)
    tmp18 = tl.load(in_ptr0 + (12 + x0), xmask)
    tmp19 = tl.load(in_ptr1 + (12 + x0), xmask)
    tmp2 = tmp0 / tmp1
    tmp3 = 0.25
    tmp4 = tmp2 * tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp8 = tmp6 / tmp7
    tmp9 = tmp8 * tmp3
    tmp10 = tl_math.exp(tmp9)
    tmp11 = tmp5 + tmp10
    tmp14 = tmp12 / tmp13
    tmp15 = tmp14 * tmp3
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp11 + tmp16
    tmp20 = tmp18 / tmp19
    tmp21 = tmp20 * tmp3
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp17 + tmp22
    tmp24 = 1e-08
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x0), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ns/cns6l44m4yb6rxd7by256qxjeq7jhymqciz2lsaeid2tbrzehqsp.py
# Topologically Sorted Source Nodes: [truediv, truediv_1, sim_matrix, add, matrix_mp2sc, mul, sum_2, log, mean, lori_mp, add_1, matrix_sc2mp_1, mul_1, sum_4, log_1, mean_1, lori_sc, mul_2, mul_3, contra_loss], Original ATen: [aten.div, aten.exp, aten.add, aten.mul, aten.sum, aten.log, aten.mean, aten.neg]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   contra_loss => add_2
#   log => log
#   log_1 => log_1
#   lori_mp => neg
#   lori_sc => neg_1
#   matrix_mp2sc => div_2
#   matrix_sc2mp_1 => div_3
#   mean => mean
#   mean_1 => mean_1
#   mul => mul_6
#   mul_1 => mul_7
#   mul_2 => mul_8
#   mul_3 => mul_9
#   sim_matrix => exp
#   sum_2 => sum_4
#   sum_4 => sum_6
#   truediv => div
#   truediv_1 => div_1
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mm, %mm_1), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, 4), kwargs = {})
#   %exp : [num_users=3] = call_function[target=torch.ops.aten.exp.default](args = (%div_1,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view, 1e-08), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %add), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, %primals_7), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_6, [-1]), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_4,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%log,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mean,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, 1e-08), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_6, %add_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_3, %primals_7), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_7, [-1]), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%sum_6,), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.default](args = (%log_1,), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%mean_1,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, 4), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, -3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
triton_per_fused_add_div_exp_log_mean_mul_neg_sum_4 = async_compile.triton('triton_per_fused_add_div_exp_log_mean_mul_neg_sum_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': (6,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_exp_log_mean_mul_neg_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 22, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_exp_log_mean_mul_neg_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (4*r0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*r0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (r0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (4*r2), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr1 + (1 + 4*r0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr3 + (1 + 4*r2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*r0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (2 + 4*r2), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr0 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (3 + 4*r0), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (3 + 4*r2), None, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr0 + (r0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr1 + (r0), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (r0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr0 + (4 + r0), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr1 + (4 + r0), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (8 + r0), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr1 + (8 + r0), None, eviction_policy='evict_last')
    tmp61 = tl.load(in_ptr0 + (12 + r0), None, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr1 + (12 + r0), None, eviction_policy='evict_last')
    tmp2 = tmp0 / tmp1
    tmp3 = 0.25
    tmp4 = tmp2 * tmp3
    tmp5 = tl_math.exp(tmp4)
    tmp7 = tmp5 / tmp6
    tmp9 = tmp7 * tmp8
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12 * tmp3
    tmp14 = tl_math.exp(tmp13)
    tmp15 = tmp14 / tmp6
    tmp17 = tmp15 * tmp16
    tmp18 = tmp9 + tmp17
    tmp21 = tmp19 / tmp20
    tmp22 = tmp21 * tmp3
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp23 / tmp6
    tmp26 = tmp24 * tmp25
    tmp27 = tmp18 + tmp26
    tmp30 = tmp28 / tmp29
    tmp31 = tmp30 * tmp3
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp32 / tmp6
    tmp35 = tmp33 * tmp34
    tmp36 = tmp27 + tmp35
    tmp39 = tmp37 / tmp38
    tmp40 = tmp39 * tmp3
    tmp41 = tl_math.exp(tmp40)
    tmp43 = tmp41 / tmp42
    tmp44 = tmp43 * tmp8
    tmp47 = tmp45 / tmp46
    tmp48 = tmp47 * tmp3
    tmp49 = tl_math.exp(tmp48)
    tmp50 = tmp49 / tmp42
    tmp51 = tmp50 * tmp16
    tmp52 = tmp44 + tmp51
    tmp55 = tmp53 / tmp54
    tmp56 = tmp55 * tmp3
    tmp57 = tl_math.exp(tmp56)
    tmp58 = tmp57 / tmp42
    tmp59 = tmp58 * tmp25
    tmp60 = tmp52 + tmp59
    tmp63 = tmp61 / tmp62
    tmp64 = tmp63 * tmp3
    tmp65 = tl_math.exp(tmp64)
    tmp66 = tmp65 / tmp42
    tmp67 = tmp66 * tmp34
    tmp68 = tmp60 + tmp67
    tmp69 = tl_math.log(tmp36)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, RBLOCK])
    tmp72 = tl.sum(tmp70, 1)[:, None]
    tmp73 = tl_math.log(tmp68)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
    tmp76 = tl.sum(tmp74, 1)[:, None]
    tmp77 = 64.0
    tmp78 = tmp72 / tmp77
    tmp79 = -tmp78
    tmp80 = 4.0
    tmp81 = tmp79 * tmp80
    tmp82 = tmp76 / tmp77
    tmp83 = -tmp82
    tmp84 = -3.0
    tmp85 = tmp83 * tmp84
    tmp86 = tmp81 + tmp85
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp86, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4), (4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4), (4, 1))
    assert_size_stride(primals_7, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_2, primals_3, reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf0)
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf0, buf1, 16, grid=grid(16), stream=stream0)
        buf2 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf1, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf2)
        buf3 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_2, primals_6, reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf3)
        del primals_1
        del primals_2
        buf4 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.elu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_elu_0.run(buf3, buf4, 16, grid=grid(16), stream=stream0)
        buf5 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_6], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_5, buf4, reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf5)
        del primals_5
        buf6 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z1_norm], Original ATen: [aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_linalg_vector_norm_1.run(buf2, buf6, 4, grid=grid(4), stream=stream0)
        buf7 = empty_strided_cuda((4, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [z2_norm], Original ATen: [aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_linalg_vector_norm_1.run(buf5, buf7, 4, grid=grid(4), stream=stream0)
        buf8 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dot_numerator], Original ATen: [aten.mm]
        extern_kernels.mm(buf2, reinterpret_tensor(buf5, (4, 4), (1, 4), 0), out=buf8)
        buf9 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [dot_denominator], Original ATen: [aten.mm]
        extern_kernels.mm(buf6, reinterpret_tensor(buf7, (1, 4), (1, 1), 0), out=buf9)
        buf10 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf8, buf9, buf10, 4, grid=grid(4), stream=stream0)
        buf13 = empty_strided_cuda((4, 1), (1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_3.run(buf8, buf9, buf13, 4, grid=grid(4), stream=stream0)
        buf12 = empty_strided_cuda((), (), torch.float32)
        buf16 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [truediv, truediv_1, sim_matrix, add, matrix_mp2sc, mul, sum_2, log, mean, lori_mp, add_1, matrix_sc2mp_1, mul_1, sum_4, log_1, mean_1, lori_sc, mul_2, mul_3, contra_loss], Original ATen: [aten.div, aten.exp, aten.add, aten.mul, aten.sum, aten.log, aten.mean, aten.neg]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_exp_log_mean_mul_neg_sum_4.run(buf16, buf8, buf9, buf10, primals_7, buf13, 1, 64, grid=grid(1), stream=stream0)
        del buf10
        del buf13
    return (buf16, primals_3, primals_6, primals_7, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
