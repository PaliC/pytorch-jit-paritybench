# AOT ID: ['4_forward']
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


# kernel path: inductor_cache/eb/cebf6dgqlx6joxacejncpqhd6ytjqt4rurhu4y2gj3hlrtulllsq.py
# Topologically Sorted Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu]
# Source node to ATen node mapping:
#   leaky_relu => gt, mul, where
# Graph fragment:
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%select, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, 0.2), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %select, %mul), kwargs = {})
#   %copy__default : [num_users=0] = call_function[target=torch.ops.aten.copy_.default](args = (%select_int, %where), kwargs = {})
triton_poi_fused_leaky_relu_0 = async_compile.triton('triton_poi_fused_leaky_relu_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_leaky_relu_0', 'mutated_arg_names': ['in_ptr0', 'out_ptr1'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_leaky_relu_0(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 > tmp1
    tmp3 = 0.2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tl.store(out_ptr1 + (x0), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/56/c56vybhd7a66doqce3qp4ewcbra2lzzsy5o7uuixiexvlrqsn5hk.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select_5, %select_6), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0), xmask)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xn/cxnxogpjiftobwhz3hmvdbosocmdylucs7xqjlzrrgfowrb6rv4y.py
# Topologically Sorted Source Nodes: [instance_norm], Original ATen: [aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   instance_norm => var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%unsqueeze_5, [0, 2, 3]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused__native_batch_norm_legit_2 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp4 = 1.0
    tmp5 = tl.where(tmp2, tmp4, tmp0)
    tmp6 = tmp3 / tmp5
    tmp7 = tl.where(tmp2, tmp1, tmp6)
    tmp9 = tmp8 == tmp1
    tmp11 = tl.where(tmp9, tmp4, tmp8)
    tmp12 = tmp10 / tmp11
    tmp13 = tl.where(tmp9, tmp1, tmp12)
    tmp14 = tmp7 + tmp13
    tmp16 = tmp15 == tmp1
    tmp18 = tl.where(tmp16, tmp4, tmp15)
    tmp19 = tmp17 / tmp18
    tmp20 = tl.where(tmp16, tmp1, tmp19)
    tmp21 = tmp14 + tmp20
    tmp23 = tmp22 == tmp1
    tmp25 = tl.where(tmp23, tmp4, tmp22)
    tmp26 = tmp24 / tmp25
    tmp27 = tl.where(tmp23, tmp1, tmp26)
    tmp28 = tmp21 + tmp27
    tmp29 = 4.0
    tmp30 = tmp28 / tmp29
    tmp31 = tmp7 - tmp30
    tmp32 = tmp31 * tmp31
    tmp33 = tmp13 - tmp30
    tmp34 = tmp33 * tmp33
    tmp35 = tmp32 + tmp34
    tmp36 = tmp20 - tmp30
    tmp37 = tmp36 * tmp36
    tmp38 = tmp35 + tmp37
    tmp39 = tmp27 - tmp30
    tmp40 = tmp39 * tmp39
    tmp41 = tmp38 + tmp40
    tmp42 = tmp41 / tmp29
    tl.store(out_ptr0 + (x0), tmp30, xmask)
    tl.store(out_ptr1 + (x0), tmp42, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rr/crr626p6ul5s5guzdkowso3f4p6isjb7ehcwsdgsei4uw4uliadl.py
# Topologically Sorted Source Nodes: [no_update_holes, output_1, new_mask, new_mask_1, instance_norm], Original ATen: [aten.eq, aten.masked_fill, aten.ones_like, aten._native_batch_norm_legit]
# Source node to ATen node mapping:
#   instance_norm => add_1, add_2, mul_2, mul_3, rsqrt, sub_1
#   new_mask => full_default_3
#   new_mask_1 => where_3
#   no_update_holes => eq
#   output_1 => full_default_2
# Graph fragment:
#   %eq : [num_users=3] = call_function[target=torch.ops.aten.eq.Scalar](args = (%squeeze_1, 0), kwargs = {})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 2, 2], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default_2, %full_default_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_5, %getitem_1), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %unsqueeze_9), kwargs = {})
triton_poi_fused__native_batch_norm_legit_eq_masked_fill_ones_like_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_eq_masked_fill_ones_like_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_eq_masked_fill_ones_like_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_eq_masked_fill_ones_like_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    x2 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp5 = tl.load(in_ptr1 + (x0), xmask)
    tmp9 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tmp3 = 1.0
    tmp4 = tl.where(tmp2, tmp1, tmp3)
    tmp6 = tl.where(tmp2, tmp3, tmp0)
    tmp7 = tmp5 / tmp6
    tmp8 = tl.where(tmp2, tmp1, tmp7)
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp10 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp19, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [leaky_relu], Original ATen: [aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_leaky_relu_0.run(primals_1, primals_1, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [output_mask], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(reinterpret_tensor(primals_1, (1, 4, 4, 4), (64, 16, 4, 1), 64), primals_3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (1, 4, 2, 2), (16, 4, 2, 1))
        del primals_3
        buf4 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(primals_1, buf4, 64, grid=grid(64), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(reinterpret_tensor(buf4, (1, 4, 4, 4), (0, 16, 4, 1), 0), primals_2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (1, 4, 2, 2), (16, 4, 2, 1))
        buf8 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        buf9 = empty_strided_cuda((1, 4, 1, 1), (4, 1, 4, 4), torch.float32)
        # Topologically Sorted Source Nodes: [instance_norm], Original ATen: [aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_2.run(buf6, buf5, buf8, buf9, 4, grid=grid(4), stream=stream0)
        buf7 = empty_strided_cuda((4, 2, 2), (4, 2, 1), torch.float32)
        buf10 = empty_strided_cuda((1, 4, 2, 2), (16, 4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [no_update_holes, output_1, new_mask, new_mask_1, instance_norm], Original ATen: [aten.eq, aten.masked_fill, aten.ones_like, aten._native_batch_norm_legit]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_eq_masked_fill_ones_like_3.run(buf6, buf5, buf8, buf9, primals_4, primals_5, buf7, buf10, 16, grid=grid(16), stream=stream0)
        del buf8
        del buf9
        del primals_5
    return (reinterpret_tensor(buf10, (4, 2, 2), (4, 2, 1), 0), buf7, primals_2, primals_4, reinterpret_tensor(buf4, (1, 4, 4, 4), (64, 16, 4, 1), 0), buf5, buf6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
