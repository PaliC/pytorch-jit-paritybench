# AOT ID: ['0_inference']
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


cpp_fused_lift_fresh_prod_0 = async_compile.cpp_pybinding(['int64_t*', 'int64_t*', 'int64_t*', 'int64_t*'], '''
#include "inductor_cache/2r/c2rnilspx43ivnzu4uieul65kx65dfhfbptbh5og4wk6rqebuxoo.h"
extern "C"  void kernel(int64_t* out_ptr0,
                       int64_t* out_ptr1,
                       int64_t* out_ptr2,
                       int64_t* out_ptr3)
{
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(16);
                        auto tmp5 = static_cast<int64_t>(4);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        tmp_acc0 = tmp_acc0 * tmp6;
                    }
                }
            }
            out_ptr0[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(16);
                        auto tmp5 = static_cast<int64_t>(4);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        tmp_acc0 = tmp_acc0 * tmp6;
                    }
                }
            }
            out_ptr1[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(16);
                        auto tmp5 = static_cast<int64_t>(4);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        tmp_acc0 = tmp_acc0 * tmp6;
                    }
                }
            }
            out_ptr2[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
    {
        {
            int64_t tmp_acc0 = 1;
            for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
            {
                {
                    {
                        auto tmp0 = x0;
                        auto tmp1 = c10::convert<int64_t>(tmp0);
                        auto tmp2 = static_cast<int64_t>(1);
                        auto tmp3 = tmp1 < tmp2;
                        auto tmp4 = static_cast<int64_t>(16);
                        auto tmp5 = static_cast<int64_t>(4);
                        auto tmp6 = tmp3 ? tmp4 : tmp5;
                        tmp_acc0 = tmp_acc0 * tmp6;
                    }
                }
            }
            out_ptr3[static_cast<int64_t>(0L)] = tmp_acc0;
        }
    }
}
''')


# kernel path: inductor_cache/cl/cclaogvu35oyigggoqrkthriayz25vsyj7zvimw5xqhn2qqenpgq.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.clone, aten.view]
# Source node to ATen node mapping:
#   x_1 => clone
#   x_2 => view
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
#   %view : [num_users=5] = call_function[target=torch.ops.aten.reshape.default](args = (%clone, [-1, 4]), kwargs = {})
triton_poi_fused_clone_view_1 = async_compile.triton('triton_poi_fused_clone_view_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_view_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (4*x1 + 16*(y0 // 4) + ((y0 % 4))), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ef/cefyzmqyf45ow5drswik26g5a4egcocnha7jmbzwvnwvdkdox2ik.py
# Topologically Sorted Source Nodes: [pow_1, sum_1, mul, sub_1, pow_2, sum_2, distance], Original ATen: [aten.pow, aten.sum, aten.mul, aten.sub, aten.add]
# Source node to ATen node mapping:
#   distance => add
#   mul => mul
#   pow_1 => pow_3
#   pow_2 => pow_4
#   sub_1 => sub_1
#   sum_1 => sum_2
#   sum_2 => sum_3
# Graph fragment:
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view, 2), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_3, [-1], True), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mm, 2), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sum_2, %mul), kwargs = {})
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%permute_1, 2), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_4, [0], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sub_1, %sum_3), kwargs = {})
triton_poi_fused_add_mul_pow_sub_sum_2 = async_compile.triton('triton_poi_fused_add_mul_pow_sub_sum_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_pow_sub_sum_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_pow_sub_sum_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp15 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp3 = tmp2 * tmp2
    tmp4 = tmp1 + tmp3
    tmp6 = tmp5 * tmp5
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8 * tmp8
    tmp10 = tmp7 + tmp9
    tmp12 = 2.0
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 - tmp13
    tmp16 = tmp15 * tmp15
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp21 = tmp20 * tmp20
    tmp22 = tmp19 + tmp21
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 + tmp24
    tmp26 = tmp14 + tmp25
    tl.store(in_out_ptr0 + (x2), tmp26, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xq/cxqstppcpyxhtzrei3wyn3ud7v2goclyrsgxcs7z2lhwlx4htrsa.py
# Topologically Sorted Source Nodes: [min_1], Original ATen: [aten.min]
# Source node to ATen node mapping:
#   min_1 => min_1
# Graph fragment:
#   %min_1 : [num_users=1] = call_function[target=torch.ops.aten.min.dim](args = (%add, -1), kwargs = {})
triton_poi_fused_min_3 = async_compile.triton('triton_poi_fused_min_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i64', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_min_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_min_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 < tmp1
    tmp3 = tmp0 == tmp1
    tmp4 = tmp0 != tmp0
    tmp5 = tmp1 != tmp1
    tmp6 = tmp4 > tmp5
    tmp7 = tmp2 | tmp6
    tmp8 = tmp4 & tmp5
    tmp9 = tmp3 | tmp8
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tl.full([1], 1, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tmp7 | tmp13
    tmp15 = tl.where(tmp14, tmp0, tmp1)
    tmp16 = tl.where(tmp14, tmp10, tmp11)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp15 == tmp17
    tmp20 = tmp15 != tmp15
    tmp21 = tmp17 != tmp17
    tmp22 = tmp20 > tmp21
    tmp23 = tmp18 | tmp22
    tmp24 = tmp20 & tmp21
    tmp25 = tmp19 | tmp24
    tmp26 = tl.full([1], 2, tl.int64)
    tmp27 = tmp16 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tmp23 | tmp28
    tmp30 = tl.where(tmp29, tmp15, tmp17)
    tmp31 = tl.where(tmp29, tmp16, tmp26)
    tmp33 = tmp30 < tmp32
    tmp34 = tmp30 == tmp32
    tmp35 = tmp30 != tmp30
    tmp36 = tmp32 != tmp32
    tmp37 = tmp35 > tmp36
    tmp38 = tmp33 | tmp37
    tmp39 = tmp35 & tmp36
    tmp40 = tmp34 | tmp39
    tmp41 = tl.full([1], 3, tl.int64)
    tmp42 = tmp31 < tmp41
    tmp43 = tmp40 & tmp42
    tmp44 = tmp38 | tmp43
    tmp45 = tl.where(tmp44, tmp30, tmp32)
    tmp46 = tl.where(tmp44, tmp31, tmp41)
    tl.store(out_ptr0 + (x0), tmp46, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/it/citqb57qqleiv7kc2w6yzrb37f2slbqb46xtbo3r64xpq6qi4rcm.py
# Topologically Sorted Source Nodes: [x_3, sub_2, norm_1, pow_3, commit_loss], Original ATen: [aten.embedding, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div]
# Source node to ATen node mapping:
#   commit_loss => div_1
#   norm_1 => pow_5, pow_6, sum_4
#   pow_3 => pow_7
#   sub_2 => sub_2
#   x_3 => embedding
# Graph fragment:
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%arg1_1, %getitem_1), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%embedding, %view), kwargs = {})
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sub_2, 2), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%pow_5, None), kwargs = {})
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%sum_4, 0.5), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%pow_6, 2), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%pow_7, %prod_1), kwargs = {})
triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4 = async_compile.triton('triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': 'i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 6), 'tt.equal_to': (5,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex // 4
    r0 = (rindex % 4)
    r2 = rindex
    tmp0 = tl.load(in_ptr0 + (r1), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr2 + (r2), None)
    tmp15 = in_ptr3
    tmp1 = tl.full([XBLOCK, RBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 4), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (r0 + 4*tmp4), None)
    tmp8 = tmp6 - tmp7
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None]
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = tmp13 * tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp14 / tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/ph/cphaitl23vnaybcdhpanqriqgd2j2b2yww2l5q4tenn4zbiaoatd.py
# Topologically Sorted Source Nodes: [x_d_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_d_1 => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
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
    tmp0 = tl.load(in_ptr0 + (x2 + 4*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.full([XBLOCK, YBLOCK], 4, tl.int32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp1 < 0
    tmp5 = tl.where(tmp4, tmp3, tmp1)
    tl.device_assert(((0 <= tmp5) & (tmp5 < 4)) | ~(ymask), "index out of bounds: 0 <= tmp5 < 4")
    tmp7 = tl.load(in_ptr2 + (x2 + 4*tmp5), xmask & ymask)
    tmp8 = tmp7 - tmp0
    tmp9 = tmp0 + tmp8
    tl.store(out_ptr0 + (y0 + 4*x2 + 16*y1), tmp9, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/yt/cytgb5i7djqw5zsrkd4nzr5mfbdtc2w6vs2psklsfxoowwnmcpvu.py
# Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.clone, aten.view]
# Source node to ATen node mapping:
#   x_5 => clone_2
#   x_6 => view_3
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %view_3 : [num_users=5] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_2, [-1, 4]), kwargs = {})
triton_poi_fused_clone_view_6 = async_compile.triton('triton_poi_fused_clone_view_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_view_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (64 + 4*x1 + 16*(y0 // 4) + ((y0 % 4))), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/k4/ck4w5zkyw35rjii6ol7rh22u5p6a5anotvravspyfztgmh3avhpe.py
# Topologically Sorted Source Nodes: [x_9, x_10], Original ATen: [aten.clone, aten.view]
# Source node to ATen node mapping:
#   x_10 => view_6
#   x_9 => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
#   %view_6 : [num_users=5] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_4, [-1, 4]), kwargs = {})
triton_poi_fused_clone_view_7 = async_compile.triton('triton_poi_fused_clone_view_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_view_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (128 + 4*x1 + 16*(y0 // 4) + ((y0 % 4))), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/dk/cdk34uizlx3xaq5un54bfbnitzmj4mdr2s2tly2fcfr5qoaprnr6.py
# Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten.clone, aten.view]
# Source node to ATen node mapping:
#   x_13 => clone_6
#   x_14 => view_9
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_9,), kwargs = {memory_format: torch.contiguous_format})
#   %view_9 : [num_users=5] = call_function[target=torch.ops.aten.reshape.default](args = (%clone_6, [-1, 4]), kwargs = {})
triton_poi_fused_clone_view_8 = async_compile.triton('triton_poi_fused_clone_view_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_view_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_view_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (192 + 4*x1 + 16*(y0 // 4) + ((y0 % 4))), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 4*y0), tmp0, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4), (4, 1))
    assert_size_stride(arg2_1, (4, 4), (4, 1))
    assert_size_stride(arg3_1, (4, 4), (4, 1))
    assert_size_stride(arg4_1, (4, 4), (4, 1))
    buf21 = empty_strided_cpu((), (), torch.int64)
    buf23 = empty_strided_cpu((), (), torch.int64)
    buf25 = empty_strided_cpu((), (), torch.int64)
    buf27 = empty_strided_cpu((), (), torch.int64)
    cpp_fused_lift_fresh_prod_0(buf21, buf23, buf25, buf27)
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.clone, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_view_1.run(arg0_1, buf0, 16, 4, grid=grid(16, 4), stream=stream0)
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(buf0, reinterpret_tensor(arg1_1, (4, 4), (1, 4), 0), out=buf1)
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [pow_1, sum_1, mul, sub_1, pow_2, sum_2, distance], Original ATen: [aten.pow, aten.sum, aten.mul, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sub_sum_2.run(buf2, buf0, arg1_1, 64, grid=grid(64), stream=stream0)
        buf3 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [min_1], Original ATen: [aten.min]
        stream0 = get_raw_stream(0)
        triton_poi_fused_min_3.run(buf2, buf3, 16, grid=grid(16), stream=stream0)
        buf20 = empty_strided_cuda((), (), torch.float32)
        buf28 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [x_3, sub_2, norm_1, pow_3, commit_loss], Original ATen: [aten.embedding, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4.run(buf28, buf3, arg1_1, buf0, buf21.item(), 1, 64, grid=grid(1), stream=stream0)
        del buf21
        buf16 = reinterpret_tensor(buf2, (4, 4, 4), (16, 4, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_d_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf0, buf3, arg1_1, buf16, 16, 4, grid=grid(16, 4), stream=stream0)
        del arg1_1
        buf4 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_5, x_6], Original ATen: [aten.clone, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_view_6.run(arg0_1, buf4, 16, 4, grid=grid(16, 4), stream=stream0)
        buf5 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf4, reinterpret_tensor(arg2_1, (4, 4), (1, 4), 0), out=buf5)
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [pow_4, sum_3, mul_1, sub_5, pow_5, sum_4, distance_1], Original ATen: [aten.pow, aten.sum, aten.mul, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sub_sum_2.run(buf6, buf4, arg2_1, 64, grid=grid(64), stream=stream0)
        buf7 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [min_2], Original ATen: [aten.min]
        stream0 = get_raw_stream(0)
        triton_poi_fused_min_3.run(buf6, buf7, 16, grid=grid(16), stream=stream0)
        buf22 = empty_strided_cuda((), (), torch.float32)
        buf29 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_7, sub_6, norm_3, pow_6, commit_loss_1], Original ATen: [aten.embedding, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4.run(buf29, buf7, arg2_1, buf4, buf23.item(), 1, 64, grid=grid(1), stream=stream0)
        del buf23
        buf17 = reinterpret_tensor(buf6, (4, 4, 4), (16, 4, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [x_d_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf4, buf7, arg2_1, buf17, 16, 4, grid=grid(16, 4), stream=stream0)
        del arg2_1
        buf8 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [x_9, x_10], Original ATen: [aten.clone, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_view_7.run(arg0_1, buf8, 16, 4, grid=grid(16, 4), stream=stream0)
        buf9 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.mm]
        extern_kernels.mm(buf8, reinterpret_tensor(arg3_1, (4, 4), (1, 4), 0), out=buf9)
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [pow_7, sum_5, mul_2, sub_9, pow_8, sum_6, distance_2], Original ATen: [aten.pow, aten.sum, aten.mul, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sub_sum_2.run(buf10, buf8, arg3_1, 64, grid=grid(64), stream=stream0)
        buf11 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [min_3], Original ATen: [aten.min]
        stream0 = get_raw_stream(0)
        triton_poi_fused_min_3.run(buf10, buf11, 16, grid=grid(16), stream=stream0)
        buf24 = empty_strided_cuda((), (), torch.float32)
        buf30 = buf24; del buf24  # reuse
        # Topologically Sorted Source Nodes: [x_11, sub_10, norm_5, pow_9, commit_loss_2], Original ATen: [aten.embedding, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4.run(buf30, buf11, arg3_1, buf8, buf25.item(), 1, 64, grid=grid(1), stream=stream0)
        del buf25
        buf18 = reinterpret_tensor(buf10, (4, 4, 4), (16, 4, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_d_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf8, buf11, arg3_1, buf18, 16, 4, grid=grid(16, 4), stream=stream0)
        del arg3_1
        buf12 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten.clone, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_view_8.run(arg0_1, buf12, 16, 4, grid=grid(16, 4), stream=stream0)
        del arg0_1
        buf13 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf12, reinterpret_tensor(arg4_1, (4, 4), (1, 4), 0), out=buf13)
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [pow_10, sum_7, mul_3, sub_13, pow_11, sum_8, distance_3], Original ATen: [aten.pow, aten.sum, aten.mul, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_pow_sub_sum_2.run(buf14, buf12, arg4_1, 64, grid=grid(64), stream=stream0)
        buf15 = empty_strided_cuda((16, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [min_4], Original ATen: [aten.min]
        stream0 = get_raw_stream(0)
        triton_poi_fused_min_3.run(buf14, buf15, 16, grid=grid(16), stream=stream0)
        buf26 = empty_strided_cuda((), (), torch.float32)
        buf31 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [x_15, sub_14, norm_7, pow_12, commit_loss_3], Original ATen: [aten.embedding, aten.sub, aten.linalg_vector_norm, aten.pow, aten.div]
        stream0 = get_raw_stream(0)
        triton_per_fused_div_embedding_linalg_vector_norm_pow_sub_4.run(buf31, buf15, arg4_1, buf12, buf27.item(), 1, 64, grid=grid(1), stream=stream0)
        del buf27
        buf19 = reinterpret_tensor(buf14, (4, 4, 4), (16, 4, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [x_d_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf12, buf15, arg4_1, buf19, 16, 4, grid=grid(16, 4), stream=stream0)
        del arg4_1
        del buf12
    return (reinterpret_tensor(buf3, (4, 4), (4, 1), 0), reinterpret_tensor(buf7, (4, 4), (4, 1), 0), reinterpret_tensor(buf11, (4, 4), (4, 1), 0), reinterpret_tensor(buf15, (4, 4), (4, 1), 0), buf16, buf17, buf18, buf19, buf28, buf29, buf30, buf31, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
