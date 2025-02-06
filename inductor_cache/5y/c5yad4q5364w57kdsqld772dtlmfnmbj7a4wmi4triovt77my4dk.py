# AOT ID: ['2_inference']
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


# kernel path: inductor_cache/6z/c6zsvyyo6qpangayibgfbbxpvkjtnszrloqn42z7ywgh7ba6isrl.py
# Topologically Sorted Source Nodes: [rvec], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   rvec => div
# Graph fragment:
#   %div : [num_users=24] = call_function[target=torch.ops.aten.div.Tensor](args = (%arg0_1, %unsqueeze), kwargs = {})
triton_poi_fused_div_0 = async_compile.triton('triton_poi_fused_div_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (32 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (48 + x0 + 64*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f6/cf6afbwhqmkooa6h32adlolvpy4zvccuqtiid7qqmnfk22r7si7l.py
# Topologically Sorted Source Nodes: [pow_1, sum_1, add, theta, pow_2, pow_3, sub, costh, mul, add_1, mul_1, sub_1, mul_2, sinth, mul_3, sub_2, mul_4, sub_3, mul_5, mul_6, add_2, mul_7, sub_4, mul_8, mul_9, add_3, pow_4, pow_5, sub_5, mul_10, add_4, mul_11, sub_6, mul_12, mul_13, sub_7, mul_14, sub_8, mul_15, mul_16, sub_9, mul_17, sub_10, mul_18, mul_19, add_5, pow_6, pow_7, sub_11, mul_20, add_6], Original ATen: [aten.pow, aten.sum, aten.add, aten.sqrt, aten.rsub, aten.cos, aten.mul, aten.sin, aten.sub]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   add_2 => add_2
#   add_3 => add_3
#   add_4 => add_4
#   add_5 => add_5
#   add_6 => add_6
#   costh => cos
#   mul => mul
#   mul_1 => mul_1
#   mul_10 => mul_10
#   mul_11 => mul_11
#   mul_12 => mul_12
#   mul_13 => mul_13
#   mul_14 => mul_14
#   mul_15 => mul_15
#   mul_16 => mul_16
#   mul_17 => mul_17
#   mul_18 => mul_18
#   mul_19 => mul_19
#   mul_2 => mul_2
#   mul_20 => mul_20
#   mul_3 => mul_3
#   mul_4 => mul_4
#   mul_5 => mul_5
#   mul_6 => mul_6
#   mul_7 => mul_7
#   mul_8 => mul_8
#   mul_9 => mul_9
#   pow_1 => pow_1
#   pow_2 => pow_2
#   pow_3 => pow_3
#   pow_4 => pow_4
#   pow_5 => pow_5
#   pow_6 => pow_6
#   pow_7 => pow_7
#   sinth => sin
#   sub => sub
#   sub_1 => sub_1
#   sub_10 => sub_10
#   sub_11 => sub_11
#   sub_2 => sub_2
#   sub_3 => sub_3
#   sub_4 => sub_4
#   sub_5 => sub_5
#   sub_6 => sub_6
#   sub_7 => sub_7
#   sub_8 => sub_8
#   sub_9 => sub_9
#   sum_1 => sum_1
#   theta => sqrt
# Graph fragment:
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%arg0_1, 2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_1, [1]), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, 1e-05), kwargs = {})
#   %sqrt : [num_users=3] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select, 2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_1, 2), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_3), kwargs = {})
#   %cos : [num_users=9] = call_function[target=torch.ops.aten.cos.default](args = (%sqrt,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %cos), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_2, %mul), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_2, %select_3), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %cos), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %sub_1), kwargs = {})
#   %sin : [num_users=6] = call_function[target=torch.ops.aten.sin.default](args = (%sqrt,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_4, %sin), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_2, %mul_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %select_6), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %cos), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %sub_3), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_7, %sin), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_8, %select_9), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %cos), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %sub_4), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_10, %sin), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_11, 2), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_12, 2), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_5), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %cos), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_4, %mul_10), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_13, %select_14), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %cos), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %sub_6), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_15, %sin), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_12, %mul_13), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_16, %select_17), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %cos), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_14, %sub_8), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_18, %sin), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_15, %mul_16), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_19, %select_20), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %cos), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_17, %sub_10), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_21, %sin), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18, %mul_19), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_22, 2), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%select_23, 2), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1.0, %pow_7), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %cos), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%pow_6, %mul_20), kwargs = {})
triton_poi_fused_add_cos_mul_pow_rsub_sin_sqrt_sub_sum_1 = async_compile.triton('triton_poi_fused_add_cos_mul_pow_rsub_sin_sqrt_sub_sum_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'out_ptr5': '*fp32', 'out_ptr6': '*fp32', 'out_ptr7': '*fp32', 'out_ptr8': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cos_mul_pow_rsub_sin_sqrt_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cos_mul_pow_rsub_sin_sqrt_sub_sum_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x1 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x1), xmask)
    tmp4 = tl.load(in_ptr1 + (x0 + 64*x1), xmask)
    tmp6 = tl.load(in_ptr1 + (16 + x0 + 64*x1), xmask)
    tmp9 = tl.load(in_ptr1 + (32 + x0 + 64*x1), xmask)
    tmp12 = tl.load(in_ptr1 + (48 + x0 + 64*x1), xmask)
    tmp21 = tl.load(in_ptr0 + (16 + x0 + 64*x1), xmask)
    tmp25 = tl.load(in_ptr0 + (32 + x0 + 64*x1), xmask)
    tmp1 = tmp0 * tmp0
    tmp2 = 1.0
    tmp3 = tmp2 - tmp1
    tmp5 = tmp4 * tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp13 = tmp12 * tmp12
    tmp14 = tmp11 + tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = tl_math.cos(tmp17)
    tmp19 = tmp3 * tmp18
    tmp20 = tmp1 + tmp19
    tmp22 = tmp0 * tmp21
    tmp23 = tmp2 - tmp18
    tmp24 = tmp22 * tmp23
    tmp26 = tl_math.sin(tmp17)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 - tmp27
    tmp29 = tmp0 * tmp25
    tmp30 = tmp29 * tmp23
    tmp31 = tmp21 * tmp26
    tmp32 = tmp30 + tmp31
    tmp33 = tmp24 + tmp27
    tmp34 = tmp21 * tmp25
    tmp35 = tmp34 * tmp23
    tmp36 = tmp0 * tmp26
    tmp37 = tmp35 - tmp36
    tmp38 = tmp30 - tmp31
    tmp39 = tmp35 + tmp36
    tmp40 = tmp21 * tmp21
    tmp41 = tmp2 - tmp40
    tmp42 = tmp41 * tmp18
    tmp43 = tmp40 + tmp42
    tmp44 = tmp25 * tmp25
    tmp45 = tmp2 - tmp44
    tmp46 = tmp45 * tmp18
    tmp47 = tmp44 + tmp46
    tl.store(out_ptr0 + (x0 + 144*x1), tmp20, xmask)
    tl.store(out_ptr1 + (x0 + 144*x1), tmp28, xmask)
    tl.store(out_ptr2 + (x0 + 144*x1), tmp32, xmask)
    tl.store(out_ptr3 + (x0 + 144*x1), tmp33, xmask)
    tl.store(out_ptr4 + (x0 + 144*x1), tmp37, xmask)
    tl.store(out_ptr5 + (x0 + 144*x1), tmp38, xmask)
    tl.store(out_ptr6 + (x0 + 144*x1), tmp39, xmask)
    tl.store(out_ptr7 + (x0 + 144*x1), tmp43, xmask)
    tl.store(out_ptr8 + (x0 + 144*x1), tmp47, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [rvec], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        buf10 = empty_strided_cuda((4, 36, 4), (144, 4, 1), torch.float32)
        buf1 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 0)  # alias
        buf2 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 16)  # alias
        buf3 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 32)  # alias
        buf4 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 48)  # alias
        buf6 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 80)  # alias
        buf7 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 96)  # alias
        buf8 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 112)  # alias
        buf5 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 64)  # alias
        buf9 = reinterpret_tensor(buf10, (4, 4, 4), (144, 4, 1), 128)  # alias
        # Topologically Sorted Source Nodes: [pow_1, sum_1, add, theta, pow_2, pow_3, sub, costh, mul, add_1, mul_1, sub_1, mul_2, sinth, mul_3, sub_2, mul_4, sub_3, mul_5, mul_6, add_2, mul_7, sub_4, mul_8, mul_9, add_3, pow_4, pow_5, sub_5, mul_10, add_4, mul_11, sub_6, mul_12, mul_13, sub_7, mul_14, sub_8, mul_15, mul_16, sub_9, mul_17, sub_10, mul_18, mul_19, add_5, pow_6, pow_7, sub_11, mul_20, add_6], Original ATen: [aten.pow, aten.sum, aten.add, aten.sqrt, aten.rsub, aten.cos, aten.mul, aten.sin, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cos_mul_pow_rsub_sin_sqrt_sub_sum_1.run(buf0, arg0_1, buf1, buf2, buf3, buf4, buf6, buf7, buf8, buf5, buf9, 64, grid=grid(64), stream=stream0)
        del arg0_1
        del buf0
    return (reinterpret_tensor(buf10, (64, 3, 3), (9, 3, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
