# AOT ID: ['24_forward']
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


# kernel path: inductor_cache/tf/ctfgguofzkybjebaggwcdnky2gqegtimfieqlfs7rmjkcyhjdyui.py
# Topologically Sorted Source Nodes: [mul_1, inner], Original ATen: [aten.mul, aten.sum]
# Source node to ATen node mapping:
#   inner => sum_2
#   mul_1 => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %primals_4), kwargs = {})
#   %sum_2 : [num_users=2] = call_function[target=torch.ops.aten.sum.default](args = (%mul_1,), kwargs = {})
triton_poi_fused_mul_sum_0 = async_compile.triton('triton_poi_fused_mul_sum_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_sum_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/5g/c5gitnynignj6nsym4w2zsfuizentrqj7c7o4s64xbiq3el5z3ge.py
# Topologically Sorted Source Nodes: [exp, add_1, log, sub, sub_1, mul_2, pow_1, sum_3, truediv, u], Original ATen: [aten.exp, aten.add, aten.log, aten.sub, aten.mul, aten.pow, aten.sum, aten.div]
# Source node to ATen node mapping:
#   add_1 => add_1
#   exp => exp
#   log => log
#   mul_2 => mul_2
#   pow_1 => pow_1
#   sub => sub
#   sub_1 => sub_1
#   sum_3 => sum_3
#   truediv => div
#   u => add_2
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sum_2,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%exp, 1), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add_1,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%log, 1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub, %sum_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %primals_1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%primals_1, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%pow_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%mul_2, %sum_3), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, %div), kwargs = {})
triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_1 = async_compile.triton('triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp9 = tl.load(in_ptr2 + (x0), xmask)
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
    tmp7 = tmp6 - tmp4
    tmp8 = tmp7 - tmp2
    tmp10 = tmp8 * tmp9
    tmp13 = tmp12 * tmp12
    tmp16 = tmp15 * tmp15
    tmp17 = tmp13 + tmp16
    tmp20 = tmp19 * tmp19
    tmp21 = tmp17 + tmp20
    tmp24 = tmp23 * tmp23
    tmp25 = tmp21 + tmp24
    tmp26 = tmp10 / tmp25
    tmp27 = tmp0 + tmp26
    tl.store(out_ptr0 + (x0), tmp27, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6i/c6izywpogqim5aza4nqk2kel4e5szaqkystmexjc7lvbioayosaj.py
# Topologically Sorted Source Nodes: [mul, sum_1, lin, mul_4, sum_4, cosh, pow_2, truediv_1, mul_5, add_4, abs_1, log_det], Original ATen: [aten.mul, aten.sum, aten.add, aten.cosh, aten.pow, aten.reciprocal, aten.abs, aten.log]
# Source node to ATen node mapping:
#   abs_1 => abs_1
#   add_4 => add_4
#   cosh => cosh
#   lin => add
#   log_det => log_1
#   mul => mul
#   mul_4 => mul_4
#   mul_5 => mul_6
#   pow_2 => pow_2
#   sum_1 => sum_1
#   sum_4 => sum_4
#   truediv_1 => mul_5, reciprocal
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %primals_2), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul, [1], True), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_1, %primals_3), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %add_2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_4,), kwargs = {})
#   %cosh : [num_users=1] = call_function[target=torch.ops.aten.cosh.default](args = (%view,), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%cosh, 2), kwargs = {})
#   %reciprocal : [num_users=1] = call_function[target=torch.ops.aten.reciprocal.default](args = (%pow_2,), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%reciprocal, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sum_4, %mul_5), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6, 1), kwargs = {})
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%add_4,), kwargs = {})
#   %log_1 : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%abs_1,), kwargs = {})
triton_poi_fused_abs_add_cosh_log_mul_pow_reciprocal_sum_2 = async_compile.triton('triton_poi_fused_abs_add_cosh_log_mul_pow_reciprocal_sum_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_abs_add_cosh_log_mul_pow_reciprocal_sum_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_abs_add_cosh_log_mul_pow_reciprocal_sum_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex // 16
    x4 = (xindex % 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x4 + 64*x2), xmask)
    tmp3 = tl.load(in_ptr1 + (16 + x4 + 64*x2), xmask)
    tmp6 = tl.load(in_ptr1 + (32 + x4 + 64*x2), xmask)
    tmp9 = tl.load(in_ptr1 + (48 + x4 + 64*x2), xmask)
    tmp12 = tl.load(in_ptr2 + (0))
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp15 = tl.load(in_ptr3 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp2 = tmp0 * tmp1
    tmp4 = tmp0 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp0 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp0 * tmp9
    tmp11 = tmp8 + tmp10
    tmp14 = tmp11 + tmp13
    tmp17 = libdevice.cosh(tmp14)
    tmp18 = tmp17 * tmp17
    tmp19 = tl.full([1], 1, tl.int32)
    tmp20 = tmp19 / tmp18
    tmp21 = 1.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp16 * tmp22
    tmp24 = tmp23 + tmp21
    tmp25 = tl_math.abs(tmp24)
    tmp26 = tl_math.log(tmp25)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5k/c5khufkcvr6wuzcvi64n3go6ua23f76y2u52rrypyi4pzmn6ntkp.py
# Topologically Sorted Source Nodes: [tanh, mul_3, z_], Original ATen: [aten.tanh, aten.mul, aten.add]
# Source node to ATen node mapping:
#   mul_3 => mul_3
#   tanh => tanh
#   z_ => add_3
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%add,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_2, %tanh), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_2, %mul_3), kwargs = {})
triton_poi_fused_add_mul_tanh_3 = async_compile.triton('triton_poi_fused_add_mul_tanh_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_tanh_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_tanh_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    x0 = (xindex % 4)
    x3 = xindex // 64
    x5 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x5 + 16*x3), xmask, eviction_policy='evict_last')
    tmp3 = libdevice.tanh(tmp2)
    tmp4 = tmp1 * tmp3
    tmp5 = tmp0 + tmp4
    tl.store(out_ptr0 + (x4), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4 = args
    args.clear()
    assert_size_stride(primals_1, (1, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (1, ), (1, ))
    assert_size_stride(primals_4, (1, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1, inner], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_0.run(primals_1, primals_4, buf1, 1, grid=grid(1), stream=stream0)
        buf2 = empty_strided_cuda((1, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [exp, add_1, log, sub, sub_1, mul_2, pow_1, sum_3, truediv, u], Original ATen: [aten.exp, aten.add, aten.log, aten.sub, aten.mul, aten.pow, aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_exp_log_mul_pow_sub_sum_1.run(primals_4, buf1, primals_1, buf2, 4, grid=grid(4), stream=stream0)
        buf4 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [mul_4, sum_4], Original ATen: [aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_sum_0.run(primals_1, buf2, buf4, 1, grid=grid(1), stream=stream0)
        buf0 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.float32)
        buf5 = empty_strided_cuda((64, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [mul, sum_1, lin, mul_4, sum_4, cosh, pow_2, truediv_1, mul_5, add_4, abs_1, log_det], Original ATen: [aten.mul, aten.sum, aten.add, aten.cosh, aten.pow, aten.reciprocal, aten.abs, aten.log]
        stream0 = get_raw_stream(0)
        triton_poi_fused_abs_add_cosh_log_mul_pow_reciprocal_sum_2.run(primals_1, primals_2, primals_3, buf4, buf0, buf5, 64, grid=grid(64), stream=stream0)
        del buf4
        buf3 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tanh, mul_3, z_], Original ATen: [aten.tanh, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_tanh_3.run(primals_2, buf2, buf0, buf3, 256, grid=grid(256), stream=stream0)
        del buf0
        del buf2
    return (buf3, buf5, primals_1, primals_2, primals_3, primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
