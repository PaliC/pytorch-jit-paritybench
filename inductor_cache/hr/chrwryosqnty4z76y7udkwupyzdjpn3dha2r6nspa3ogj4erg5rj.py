# AOT ID: ['19_forward']
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


cpp_fused_lift_fresh_prod_0 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(4);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp8;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


# kernel path: inductor_cache/z2/cz2dhrzgsrq6utfagwo432drvb2co66cqwzbi2w2bk4ghemy2kmx.py
# Topologically Sorted Source Nodes: [var_mean, mul, tensor, max_1, rsqrt, scale, shift, mul_3, sub], Original ATen: [aten.var_mean, aten.mul, aten.lift_fresh, aten.maximum, aten.rsqrt, aten.sub]
# Source node to ATen node mapping:
#   max_1 => maximum
#   mul => mul
#   mul_3 => mul_3
#   rsqrt => rsqrt
#   scale => mul_1
#   shift => mul_2
#   sub => sub
#   tensor => full_default
#   var_mean => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [1, 2, 3]), kwargs = {correction: 1, keepdim: True})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem, %prod), kwargs = {})
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([], 9.999999747378752e-05), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %maximum : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%mul, %full_default), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%maximum,), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt, %primals_2), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_1, %mul_1), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %mul_1), kwargs = {})
#   %sub : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %mul_2), kwargs = {})
triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_1 = async_compile.triton('triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': 'i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp19 = in_ptr1
    tmp25 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 63.0
    tmp18 = tmp16 / tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = 9.999999747378752e-05
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = libdevice.rsqrt(tmp23)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp0 * tmp26
    tmp28 = tmp10 * tmp26
    tmp29 = tmp27 - tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, xmask)
    tl.store(out_ptr1 + (r1 + 64*x0), tmp29, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


cpp_fused_lift_fresh_prod_2 = async_compile.cpp_pybinding(['int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(3L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(2);
                        auto tmp5 = tmp1 < tmp4;
                        auto tmp6 = static_cast<int64_t>(4);
                        auto tmp7 = tmp5 ? tmp6 : tmp6;
                        auto tmp8 = tmp3 ? tmp6 : tmp7;
                        tmp_acc0 = tmp_acc0 * tmp8;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


# kernel path: inductor_cache/mm/cmmsurdoho5dfdzki6rezxi4k3lmjxhbgaphlphueessorqhvdyh.py
# Topologically Sorted Source Nodes: [tensor, var_mean_1, mul_4, max_2, rsqrt_1, scale_1, shift_1, mul_7, sub_1], Original ATen: [aten.lift_fresh, aten.var_mean, aten.mul, aten.maximum, aten.rsqrt, aten.sub, aten.eq, aten.lt]
# Source node to ATen node mapping:
#   max_2 => maximum_1
#   mul_4 => mul_4
#   mul_7 => mul_7
#   rsqrt_1 => rsqrt_1
#   scale_1 => mul_5
#   shift_1 => mul_6
#   sub_1 => sub_1
#   tensor => full_default
#   var_mean_1 => var_mean_1
# Graph fragment:
#   %full_default : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([], 9.999999747378752e-05), kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_5, [1, 2, 3]), kwargs = {correction: 1, keepdim: True})
#   %mul_4 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_2, %prod_1), kwargs = {})
#   %maximum_1 : [num_users=1] = call_function[target=torch.ops.aten.maximum.default](args = (%mul_4, %full_default), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%maximum_1,), kwargs = {})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%rsqrt_1, %primals_6), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_3, %mul_5), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_5, %mul_5), kwargs = {})
#   %sub_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_7, %mul_6), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Tensor](args = (%mul_4, %full_default), kwargs = {})
#   %lt : [num_users=1] = call_function[target=torch.ops.aten.lt.Tensor](args = (%mul_4, %full_default), kwargs = {})
triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_3 = async_compile.triton('triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 4, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': 'i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*i1', 'out_ptr4': '*i1', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp19 = in_ptr1
    tmp27 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 63.0
    tmp18 = tmp16 / tmp17
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = 9.999999747378752e-05
    tmp23 = triton_helpers.maximum(tmp21, tmp22)
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp21 == tmp22
    tmp26 = tmp21 < tmp22
    tmp28 = tmp24 * tmp27
    tmp29 = tmp0 * tmp28
    tmp30 = tmp10 * tmp28
    tmp31 = tmp29 - tmp30
    tl.store(out_ptr2 + (x0), tmp24, xmask)
    tl.store(out_ptr3 + (x0), tmp25, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr5 + (r1 + 64*x0), tmp31, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xq/cxqoiq7ttamccpi745i5s3c7vgdn2urv43yjeaidcpvfeilk5acc.py
# Topologically Sorted Source Nodes: [signal, conv_transpose2d_1, gate, mul_8, mul_9], Original ATen: [aten.convolution, aten.sigmoid, aten.mul]
# Source node to ATen node mapping:
#   conv_transpose2d_1 => convolution_1
#   gate => sigmoid
#   mul_8 => mul_8
#   mul_9 => mul_9
#   signal => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_4, %sub, %primals_3, [1, 1], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_4, %sub_1, %primals_7, [1, 1], [1, 1], [1, 1], True, [0, 0], 1), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convolution_1,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, %sigmoid), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 1.8), kwargs = {})
triton_poi_fused_convolution_mul_sigmoid_4 = async_compile.triton('triton_poi_fused_convolution_mul_sigmoid_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_sigmoid_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_mul_sigmoid_4(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 25) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_out_ptr1 + (x3), xmask)
    tmp4 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 1.8
    tmp9 = tmp7 * tmp8
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(in_out_ptr1 + (x3), tmp5, xmask)
    tl.store(out_ptr0 + (x3), tmp9, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_5, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_6, (4, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_7, (4, ), (1, ))
    buf0 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_0(buf0)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((4, 1, 1, 1), (1, 4, 4, 4), torch.float32)
        buf4 = reinterpret_tensor(buf2, (4, 1, 1, 1), (1, 1, 1, 1), 0); del buf2  # reuse
        buf5 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [var_mean, mul, tensor, max_1, rsqrt, scale, shift, mul_3, sub], Original ATen: [aten.var_mean, aten.mul, aten.lift_fresh, aten.maximum, aten.rsqrt, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused_lift_fresh_maximum_mul_rsqrt_sub_var_mean_1.run(buf4, primals_1, buf0.item(), primals_2, buf1, buf5, 4, 64, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [signal], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(primals_4, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 4, 5, 5), (100, 25, 5, 1))
    buf8 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_2(buf8)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf9 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf12 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.float32)
        buf17 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf18 = empty_strided_cuda((4, 1, 1, 1), (1, 1, 1, 1), torch.bool)
        buf13 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensor, var_mean_1, mul_4, max_2, rsqrt_1, scale_1, shift_1, mul_7, sub_1], Original ATen: [aten.lift_fresh, aten.var_mean, aten.mul, aten.maximum, aten.rsqrt, aten.sub, aten.eq, aten.lt]
        stream0 = get_raw_stream(0)
        triton_per_fused_eq_lift_fresh_lt_maximum_mul_rsqrt_sub_var_mean_3.run(primals_5, buf8.item(), primals_6, buf9, buf12, buf17, buf18, buf13, 4, 64, grid=grid(4), stream=stream0)
        # Topologically Sorted Source Nodes: [conv_transpose2d_1], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(primals_4, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=True, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 4, 5, 5), (100, 25, 5, 1))
        buf7 = buf6; del buf6  # reuse
        buf15 = buf14; del buf14  # reuse
        buf16 = empty_strided_cuda((4, 4, 5, 5), (100, 25, 5, 1), torch.float32)
        # Topologically Sorted Source Nodes: [signal, conv_transpose2d_1, gate, mul_8, mul_9], Original ATen: [aten.convolution, aten.sigmoid, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_mul_sigmoid_4.run(buf7, buf15, primals_3, primals_7, buf16, 400, grid=grid(400), stream=stream0)
        del primals_3
        del primals_7
    return (buf16, primals_1, primals_2, primals_4, primals_5, primals_6, buf0, buf1, buf4, buf5, buf7, buf8, buf9, buf12, buf13, buf15, buf17, buf18, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
