# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/e6/ce65yh7wqr7zxbuqvhfncfwkthnza7cbuy33nkgbp7qx2vhncudo.py
# Topologically Sorted Source Nodes: [fm_t_pooled, fm_s_norm, add, fm_s, fm_t_norm, add_1, fm_t, sub, pow_1, mean], Original ATen: [aten.mean, aten.linalg_vector_norm, aten.add, aten.div, aten.sub, aten.pow]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   fm_s => div_1
#   fm_s_norm => pow_1, pow_2, sum_2
#   fm_t => div_2
#   fm_t_norm => pow_3, pow_4, sum_3
#   fm_t_pooled => mean
#   mean => mean_1
#   pow_1 => pow_5
#   sub => sub
# Graph fragment:
#   %mean : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%primals_1, [-1, -2], True), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_6, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [2, 3], True), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_2, 0.5), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, 1e-06), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_6, %add), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [2, 3], True), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_3, 0.5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_4, 1e-06), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, %add_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%div_1, %div_2), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [2, 3]), kwargs = {})
triton_per_fused_add_div_linalg_vector_norm_mean_pow_sub_0 = async_compile.triton('triton_per_fused_add_div_linalg_vector_norm_mean_pow_sub_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_linalg_vector_norm_mean_pow_sub_0', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_linalg_vector_norm_mean_pow_sub_0(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (r1 + 16*x0), xmask, other=0.0)
    tmp5 = tl.load(in_ptr1 + (r1 + 16*x0), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp6 = tmp5 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    tmp9 = tl.where(xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tmp0 * tmp0
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    tmp14 = tl.where(xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = libdevice.sqrt(tmp10)
    tmp17 = 1e-06
    tmp18 = tmp16 + tmp17
    tmp19 = tmp5 / tmp18
    tmp20 = libdevice.sqrt(tmp15)
    tmp21 = tmp20 + tmp17
    tmp22 = tmp0 / tmp21
    tmp23 = tmp19 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp27 = tl.where(xmask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None]
    tmp29 = 16.0
    tmp30 = tmp4 / tmp29
    tmp31 = tmp28 / tmp29
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp30, xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cr/ccrwvjqj4j23llv5yzr5heeybdhdkdxbiazz3ji7ru5njqitgk3q.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
# Source node to ATen node mapping:
#   input_1 => convolution
#   input_2 => relu
# Graph fragment:
#   %convolution : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%mean, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
triton_poi_fused_convolution_relu_1 = async_compile.triton('triton_poi_fused_convolution_relu_1', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_relu_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tm/ctm5fh7ijf4dp5bu4sjze3ylrq73qznnz5a55atvan2a2qthkkwi.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_3 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%relu, %primals_4, %primals_5, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_2 = async_compile.triton('triton_poi_fused_convolution_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/yc/cycyf5pwikyt3uqhes5lezx2cnhv4ota7tpzly4otrbkwosl3olo.py
# Topologically Sorted Source Nodes: [rho, sum_1, rho_1, loss, sum_2, loss_1], Original ATen: [aten.sigmoid, aten.sum, aten.div, aten.mul, aten.mean]
# Source node to ATen node mapping:
#   loss => mul
#   loss_1 => mean_2
#   rho => sigmoid
#   rho_1 => div
#   sum_1 => sum_1
#   sum_2 => sum_4
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%squeeze,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%sigmoid, [1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%sigmoid, %sum_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %mean_1), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1]), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%sum_4, [0]), kwargs = {})
triton_poi_fused_div_mean_mul_sigmoid_sum_3 = async_compile.triton('triton_poi_fused_div_mean_mul_sigmoid_sum_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_mean_mul_sigmoid_sum_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 32, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_mean_mul_sigmoid_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (2))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (3))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp16 = tl.load(in_ptr1 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_ptr1 + (1))
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK])
    tmp25 = tl.load(in_ptr1 + (2))
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp30 = tl.load(in_ptr1 + (3))
    tmp31 = tl.broadcast_to(tmp30, [XBLOCK])
    tmp34 = tl.load(in_ptr0 + (4))
    tmp35 = tl.broadcast_to(tmp34, [XBLOCK])
    tmp37 = tl.load(in_ptr0 + (5))
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK])
    tmp41 = tl.load(in_ptr0 + (6))
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK])
    tmp45 = tl.load(in_ptr0 + (7))
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK])
    tmp50 = tl.load(in_ptr1 + (4))
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK])
    tmp54 = tl.load(in_ptr1 + (5))
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK])
    tmp59 = tl.load(in_ptr1 + (6))
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK])
    tmp64 = tl.load(in_ptr1 + (7))
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK])
    tmp69 = tl.load(in_ptr0 + (8))
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp72 = tl.load(in_ptr0 + (9))
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK])
    tmp76 = tl.load(in_ptr0 + (10))
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK])
    tmp80 = tl.load(in_ptr0 + (11))
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK])
    tmp85 = tl.load(in_ptr1 + (8))
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK])
    tmp89 = tl.load(in_ptr1 + (9))
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK])
    tmp94 = tl.load(in_ptr1 + (10))
    tmp95 = tl.broadcast_to(tmp94, [XBLOCK])
    tmp99 = tl.load(in_ptr1 + (11))
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK])
    tmp104 = tl.load(in_ptr0 + (12))
    tmp105 = tl.broadcast_to(tmp104, [XBLOCK])
    tmp107 = tl.load(in_ptr0 + (13))
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK])
    tmp111 = tl.load(in_ptr0 + (14))
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK])
    tmp115 = tl.load(in_ptr0 + (15))
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK])
    tmp120 = tl.load(in_ptr1 + (12))
    tmp121 = tl.broadcast_to(tmp120, [XBLOCK])
    tmp124 = tl.load(in_ptr1 + (13))
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK])
    tmp129 = tl.load(in_ptr1 + (14))
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK])
    tmp134 = tl.load(in_ptr1 + (15))
    tmp135 = tl.broadcast_to(tmp134, [XBLOCK])
    tmp2 = tl.sigmoid(tmp1)
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp2 + tmp5
    tmp9 = tl.sigmoid(tmp8)
    tmp10 = tmp6 + tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp10 + tmp13
    tmp15 = tmp2 / tmp14
    tmp18 = tmp15 * tmp17
    tmp19 = tmp5 / tmp14
    tmp22 = tmp19 * tmp21
    tmp23 = tmp18 + tmp22
    tmp24 = tmp9 / tmp14
    tmp27 = tmp24 * tmp26
    tmp28 = tmp23 + tmp27
    tmp29 = tmp13 / tmp14
    tmp32 = tmp29 * tmp31
    tmp33 = tmp28 + tmp32
    tmp36 = tl.sigmoid(tmp35)
    tmp39 = tl.sigmoid(tmp38)
    tmp40 = tmp36 + tmp39
    tmp43 = tl.sigmoid(tmp42)
    tmp44 = tmp40 + tmp43
    tmp47 = tl.sigmoid(tmp46)
    tmp48 = tmp44 + tmp47
    tmp49 = tmp36 / tmp48
    tmp52 = tmp49 * tmp51
    tmp53 = tmp39 / tmp48
    tmp56 = tmp53 * tmp55
    tmp57 = tmp52 + tmp56
    tmp58 = tmp43 / tmp48
    tmp61 = tmp58 * tmp60
    tmp62 = tmp57 + tmp61
    tmp63 = tmp47 / tmp48
    tmp66 = tmp63 * tmp65
    tmp67 = tmp62 + tmp66
    tmp68 = tmp33 + tmp67
    tmp71 = tl.sigmoid(tmp70)
    tmp74 = tl.sigmoid(tmp73)
    tmp75 = tmp71 + tmp74
    tmp78 = tl.sigmoid(tmp77)
    tmp79 = tmp75 + tmp78
    tmp82 = tl.sigmoid(tmp81)
    tmp83 = tmp79 + tmp82
    tmp84 = tmp71 / tmp83
    tmp87 = tmp84 * tmp86
    tmp88 = tmp74 / tmp83
    tmp91 = tmp88 * tmp90
    tmp92 = tmp87 + tmp91
    tmp93 = tmp78 / tmp83
    tmp96 = tmp93 * tmp95
    tmp97 = tmp92 + tmp96
    tmp98 = tmp82 / tmp83
    tmp101 = tmp98 * tmp100
    tmp102 = tmp97 + tmp101
    tmp103 = tmp68 + tmp102
    tmp106 = tl.sigmoid(tmp105)
    tmp109 = tl.sigmoid(tmp108)
    tmp110 = tmp106 + tmp109
    tmp113 = tl.sigmoid(tmp112)
    tmp114 = tmp110 + tmp113
    tmp117 = tl.sigmoid(tmp116)
    tmp118 = tmp114 + tmp117
    tmp119 = tmp106 / tmp118
    tmp122 = tmp119 * tmp121
    tmp123 = tmp109 / tmp118
    tmp126 = tmp123 * tmp125
    tmp127 = tmp122 + tmp126
    tmp128 = tmp113 / tmp118
    tmp131 = tmp128 * tmp130
    tmp132 = tmp127 + tmp131
    tmp133 = tmp117 / tmp118
    tmp136 = tmp133 * tmp135
    tmp137 = tmp132 + tmp136
    tmp138 = tmp103 + tmp137
    tmp139 = 4.0
    tmp140 = tmp138 / tmp139
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp140, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (16, 4, 1, 1), (4, 1, 1, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (4, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf6 = empty_strided_cuda((4, 4, 1, 1), (4, 1, 16, 16), torch.float32)
        buf8 = reinterpret_tensor(buf6, (4, 4), (4, 1), 0); del buf6  # reuse
        buf1 = reinterpret_tensor(buf0, (4, 4, 1, 1), (4, 1, 1, 1), 0); del buf0  # reuse
        buf9 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [fm_t_pooled, fm_s_norm, add, fm_s, fm_t_norm, add_1, fm_t, sub, pow_1, mean], Original ATen: [aten.mean, aten.linalg_vector_norm, aten.add, aten.div, aten.sub, aten.pow]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_linalg_vector_norm_mean_pow_sub_0.run(buf9, buf1, primals_1, primals_6, 16, 16, grid=grid(16), stream=stream0)
        del primals_1
        del primals_6
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 16, 1, 1), (16, 1, 1, 1))
        buf3 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten.convolution, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_relu_1.run(buf3, primals_3, 64, grid=grid(64), stream=stream0)
        del primals_3
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 4, 1, 1), (4, 1, 1, 1))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_2.run(buf5, primals_5, 16, grid=grid(16), stream=stream0)
        del primals_5
        buf10 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [rho, sum_1, rho_1, loss, sum_2, loss_1], Original ATen: [aten.sigmoid, aten.sum, aten.div, aten.mul, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_mean_mul_sigmoid_sum_3.run(buf5, buf9, buf10, 1, grid=grid(1), stream=stream0)
    return (buf10, primals_2, primals_4, buf1, buf3, buf5, buf9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 4, 1, 1), (4, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
