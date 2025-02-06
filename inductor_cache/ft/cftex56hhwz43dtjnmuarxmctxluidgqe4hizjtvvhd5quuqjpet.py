# AOT ID: ['3_forward']
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


# kernel path: inductor_cache/6f/c6frzadhlnjuumu7utlwuugoictgvhlyx5umxkxgb2uwscscekwu.py
# Topologically Sorted Source Nodes: [clamp], Original ATen: [aten.clamp]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min
# Graph fragment:
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_3, 1.01), kwargs = {})
#   %clamp_max : [num_users=4] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused_clamp_0 = async_compile.triton('triton_poi_fused_clamp_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = 1.01
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = 6.0
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/sb/csb2i5m65sdjnhu74xotefhfekzeyvhcqq2slbavcgnmzzoeywkt.py
# Topologically Sorted Source Nodes: [clamp_1], Original ATen: [aten.clamp]
# Source node to ATen node mapping:
#   clamp_1 => clamp_max_1, clamp_min_1
# Graph fragment:
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_4, 0.1), kwargs = {})
#   %clamp_max_1 : [num_users=5] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 3.0), kwargs = {})
triton_poi_fused_clamp_1 = async_compile.triton('triton_poi_fused_clamp_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = 0.1
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = 3.0
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/zd/czdvef3cloz2ogwz5qazjvvdvw7k7l4gs7slqzzz4f4zuin57ek7.py
# Topologically Sorted Source Nodes: [clamp_2], Original ATen: [aten.clamp]
# Source node to ATen node mapping:
#   clamp_2 => clamp_max_2, clamp_min_2
# Graph fragment:
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%primals_5, 0.5), kwargs = {})
#   %clamp_max_2 : [num_users=3] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 2.0), kwargs = {})
triton_poi_fused_clamp_2 = async_compile.triton('triton_poi_fused_clamp_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = 0.5
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = 2.0
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp5, None)
''', device_str='cuda')


# kernel path: inductor_cache/og/cogra645a5pdtymj2o4cwhux4ludhg6njvs4z5kwedvluhvi2loc.py
# Topologically Sorted Source Nodes: [lowerThanMu, largerThanMu, neg, sub, pow_1, mul, exp, leftValuesActiv, masked_fill_, sub_1, neg_1, mul_2, exp_1, mul_3, rightValueActiv, masked_fill__1, output, relu, maskUpdate], Original ATen: [aten.lt, aten.ge, aten.neg, aten.sub, aten.pow, aten.mul, aten.exp, aten.masked_fill, aten.add, aten.relu]
# Source node to ATen node mapping:
#   exp => exp
#   exp_1 => exp_1
#   largerThanMu => ge
#   leftValuesActiv => mul_1
#   lowerThanMu => lt
#   maskUpdate => pow_3
#   masked_fill_ => full_default, where
#   masked_fill__1 => where_1
#   mul => mul
#   mul_2 => mul_2
#   mul_3 => mul_3
#   neg => neg
#   neg_1 => neg_1
#   output => add_1
#   pow_1 => pow_1
#   relu => relu
#   rightValueActiv => add
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %lt : [num_users=2] = call_function[target=torch.ops.aten.lt.Tensor](args = (%convolution, %clamp_max_1), kwargs = {})
#   %ge : [num_users=2] = call_function[target=torch.ops.aten.ge.Tensor](args = (%convolution, %clamp_max_1), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%clamp_max_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %clamp_max_1), kwargs = {})
#   %pow_1 : [num_users=2] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub, 2), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %pow_1), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%clamp_max, %exp), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ge, %full_default, %mul_1), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clamp_max, 1), kwargs = {})
#   %neg_1 : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%clamp_max_3,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg_1, %pow_1), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2,), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %exp_1), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, 1), kwargs = {})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%lt, %full_default, %add), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, %where_1), kwargs = {})
#   %relu : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%convolution,), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%relu, 0.8), kwargs = {})
triton_poi_fused_add_exp_ge_lt_masked_fill_mul_neg_pow_relu_sub_3 = async_compile.triton('triton_poi_fused_add_exp_ge_lt_masked_fill_mul_neg_pow_relu_sub_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*i1', 'out_ptr1': '*i1', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_exp_ge_lt_masked_fill_mul_neg_pow_relu_sub_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_exp_ge_lt_masked_fill_mul_neg_pow_relu_sub_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp5 = tl.load(in_ptr2 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.load(in_ptr3 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp19 = tl.load(in_ptr4 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp3 = tmp0 < tmp2
    tmp4 = tmp0 >= tmp2
    tmp9 = -tmp8
    tmp10 = tmp0 - tmp2
    tmp11 = tmp10 * tmp10
    tmp12 = tmp9 * tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tmp6 * tmp13
    tmp15 = 0.0
    tmp16 = tl.where(tmp4, tmp15, tmp14)
    tmp17 = 1.0
    tmp18 = tmp6 - tmp17
    tmp21 = -tmp20
    tmp22 = tmp21 * tmp11
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp18 * tmp23
    tmp25 = tmp24 + tmp17
    tmp26 = tl.where(tmp3, tmp15, tmp25)
    tmp27 = tmp16 + tmp26
    tmp28 = tl.full([1], 0, tl.int32)
    tmp29 = triton_helpers.maximum(tmp28, tmp0)
    tmp30 = 0.8
    tmp31 = libdevice.pow(tmp29, tmp30)
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp27, xmask)
    tl.store(out_ptr3 + (x0), tmp31, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (), ())
    assert_size_stride(primals_4, (), ())
    assert_size_stride(primals_5, (), ())
    assert_size_stride(primals_6, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [maskFeatures], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_2, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 4, 2, 2), (16, 4, 2, 1))
        buf1 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [clamp], Original ATen: [aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_0.run(primals_3, buf1, 1, grid=grid(1), stream=stream0)
        buf2 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [clamp_1], Original ATen: [aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_1.run(primals_4, buf2, 1, grid=grid(1), stream=stream0)
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [clamp_2], Original ATen: [aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_2.run(primals_5, buf3, 1, grid=grid(1), stream=stream0)
        buf4 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [clamp_3], Original ATen: [aten.clamp]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_2.run(primals_6, buf4, 1, grid=grid(1), stream=stream0)
        buf5 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.bool)
        buf6 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.bool)
        buf7 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        buf8 = empty_strided_cuda((4, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lowerThanMu, largerThanMu, neg, sub, pow_1, mul, exp, leftValuesActiv, masked_fill_, sub_1, neg_1, mul_2, exp_1, mul_3, rightValueActiv, masked_fill__1, output, relu, maskUpdate], Original ATen: [aten.lt, aten.ge, aten.neg, aten.sub, aten.pow, aten.mul, aten.exp, aten.masked_fill, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_exp_ge_lt_masked_fill_mul_neg_pow_relu_sub_3.run(buf0, buf2, buf1, buf3, buf4, buf5, buf6, buf7, buf8, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf9 = torch.ops.aten.set_.source_Tensor(primals_3, buf1)
        assert_size_stride(buf9, (), ())
        del primals_3
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf19 = torch.ops.aten.set_.source_Tensor(primals_4, buf2)
        assert_size_stride(buf19, (), ())
        del primals_4
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf29 = torch.ops.aten.set_.source_Tensor(primals_5, buf3)
        assert_size_stride(buf29, (), ())
        del primals_5
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf34 = torch.ops.aten.set_.source_Tensor(primals_6, buf4)
        assert_size_stride(buf34, (), ())
        del primals_6
    return (buf7, buf8, primals_1, primals_2, buf0, buf1, buf2, buf3, buf4, buf5, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
