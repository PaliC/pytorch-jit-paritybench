# AOT ID: ['17_forward']
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


# kernel path: inductor_cache/gv/cgvbjityna6t23njugixc4my37za37kpi4fvkvnj5sn7g7aj3vkk.py
# Topologically Sorted Source Nodes: [scale, mul, sub, add_, div_, clamp_, round_, mul_, sub_1, add__1], Original ATen: [aten.div, aten.mul, aten.sub, aten.add, aten.clamp, aten.round]
# Source node to ATen node mapping:
#   add_ => add
#   add__1 => add_1
#   clamp_ => clamp_max, clamp_min
#   div_ => div_1
#   mul => mul
#   mul_ => mul_1
#   round_ => round_1
#   scale => div
#   sub => sub
#   sub_1 => sub_1
# Graph fragment:
#   %div : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%primals_1, 255.0), kwargs = {})
#   %mul : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, 0.0), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul, %primals_2), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %sub), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add, %div), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_1, 0.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 255.0), kwargs = {})
#   %round_1 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%clamp_max,), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%round_1, %div), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %mul), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %sub_1), kwargs = {})
triton_poi_fused_add_clamp_div_mul_round_sub_0 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_round_sub_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_round_sub_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_round_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp3 = 0.00392156862745098
    tmp4 = tmp2 * tmp3
    tmp5 = 0.0
    tmp6 = tmp4 * tmp5
    tmp9 = tmp6 - tmp8
    tmp10 = tmp0 + tmp9
    tmp11 = tmp10 / tmp4
    tmp12 = triton_helpers.maximum(tmp11, tmp5)
    tmp13 = 255.0
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = libdevice.nearbyint(tmp14)
    tmp16 = tmp15 * tmp4
    tmp17 = tmp8 - tmp6
    tmp18 = tmp16 + tmp17
    tl.store(out_ptr0 + (x0), tmp18, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mz/cmzukgxu7ojfsjtz7y55wbhielw4tql5kiibzguyrx5dwyxbr2dm.py
# Topologically Sorted Source Nodes: [range_values, qweight], Original ATen: [aten.sub, aten.div, aten.mul, aten.add, aten.clamp, aten.round]
# Source node to ATen node mapping:
#   qweight => add_2, add_3, clamp_max_1, clamp_min_1, div_2, div_3, mul_3, mul_4, round_2, sub_3, sub_4
#   range_values => sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %view), kwargs = {})
#   %div_2 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_2, 255.0), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_2, 0.0), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %view), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_4, %sub_3), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_2, %div_2), kwargs = {})
#   %clamp_min_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_3, 0.0), kwargs = {})
#   %clamp_max_1 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_1, 255.0), kwargs = {})
#   %round_2 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%clamp_max_1,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%round_2, %div_2), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view, %mul_3), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %sub_4), kwargs = {})
triton_poi_fused_add_clamp_div_mul_round_sub_1 = async_compile.triton('triton_poi_fused_add_clamp_div_mul_round_sub_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mul_round_sub_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mul_round_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = triton_helpers.minimum(tmp1, tmp2)
    tmp9 = triton_helpers.minimum(tmp8, tmp4)
    tmp10 = triton_helpers.minimum(tmp9, tmp6)
    tmp11 = tmp7 - tmp10
    tmp12 = 0.00392156862745098
    tmp13 = tmp11 * tmp12
    tmp14 = 0.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15 - tmp10
    tmp17 = tmp0 + tmp16
    tmp18 = tmp17 / tmp13
    tmp19 = triton_helpers.maximum(tmp18, tmp14)
    tmp20 = 255.0
    tmp21 = triton_helpers.minimum(tmp19, tmp20)
    tmp22 = libdevice.nearbyint(tmp21)
    tmp23 = tmp22 * tmp13
    tmp24 = tmp10 - tmp15
    tmp25 = tmp23 + tmp24
    tl.store(out_ptr0 + (x2), tmp25, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/63/c63sc4w2gtlxttwtqlw6vqvwr3mzftjr6holjpczhnmbquiumriq.py
# Topologically Sorted Source Nodes: [qbias], Original ATen: [aten.mean, aten.sub, aten.div, aten.mul, aten.add, aten.clamp, aten.round]
# Source node to ATen node mapping:
#   qbias => add_4, add_5, clamp_max_2, clamp_min_2, div_4, div_5, mean, mean_1, mul_6, mul_7, round_3, sub_5, sub_6, sub_7
# Graph fragment:
#   %mean : [num_users=3] = call_function[target=torch.ops.aten.mean.dim](args = (%view_2, [0]), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%view_3, [0]), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean_1, %mean), kwargs = {})
#   %div_4 : [num_users=3] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub_5, 65535.0), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_4, 0.0), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, %mean), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_5, %sub_6), kwargs = {})
#   %div_5 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_4, %div_4), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%div_5, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 65535.0), kwargs = {})
#   %round_3 : [num_users=1] = call_function[target=torch.ops.aten.round.default](args = (%clamp_max_2,), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%round_3, %div_4), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mean, %mul_6), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %sub_7), kwargs = {})
triton_poi_fused_add_clamp_div_mean_mul_round_sub_2 = async_compile.triton('triton_poi_fused_add_clamp_div_mean_mul_round_sub_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_clamp_div_mean_mul_round_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_clamp_div_mean_mul_round_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (2))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (3))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp5 = triton_helpers.maximum(tmp2, tmp4)
    tmp8 = triton_helpers.maximum(tmp5, tmp7)
    tmp11 = triton_helpers.maximum(tmp8, tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 / tmp12
    tmp14 = triton_helpers.minimum(tmp2, tmp4)
    tmp15 = triton_helpers.minimum(tmp14, tmp7)
    tmp16 = triton_helpers.minimum(tmp15, tmp10)
    tmp17 = tmp16 / tmp12
    tmp18 = tmp13 - tmp17
    tmp19 = 1.5259021896696422e-05
    tmp20 = tmp18 * tmp19
    tmp21 = 0.0
    tmp22 = tmp20 * tmp21
    tmp23 = tmp22 - tmp17
    tmp24 = tmp0 + tmp23
    tmp25 = tmp24 / tmp20
    tmp26 = triton_helpers.maximum(tmp25, tmp21)
    tmp27 = 65535.0
    tmp28 = triton_helpers.minimum(tmp26, tmp27)
    tmp29 = libdevice.nearbyint(tmp28)
    tmp30 = tmp29 * tmp20
    tmp31 = tmp17 - tmp22
    tmp32 = tmp30 + tmp31
    tl.store(out_ptr0 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vw/cvwjm34abuxcqwzn67cgqxqvuhsl5eogorovqvwamuwuvivudh32.py
# Topologically Sorted Source Nodes: [add, output_1], Original ATen: [aten.add, aten.sub]
# Source node to ATen node mapping:
#   add => add_6
#   output_1 => sub_8
# Graph fragment:
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_6), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %view_6), kwargs = {})
triton_poi_fused_add_sub_3 = async_compile.triton('triton_poi_fused_add_sub_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_sub_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_sub_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 + tmp2
    tmp4 = tmp3 - tmp2
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (1, ), (1, ))
    assert_size_stride(primals_2, (1, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [scale, mul, sub, add_, div_, clamp_, round_, mul_, sub_1, add__1], Original ATen: [aten.div, aten.mul, aten.sub, aten.add, aten.clamp, aten.round]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_round_sub_0.run(primals_3, primals_1, primals_2, buf0, 256, grid=grid(256), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [range_values, qweight], Original ATen: [aten.sub, aten.div, aten.mul, aten.add, aten.clamp, aten.round]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mul_round_sub_1.run(primals_4, buf1, 16, grid=grid(16), stream=stream0)
        del primals_4
        buf2 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (64, 4), (4, 1), 0), reinterpret_tensor(buf1, (4, 4), (1, 4), 0), out=buf2)
        del buf1
        buf3 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [qbias], Original ATen: [aten.mean, aten.sub, aten.div, aten.mul, aten.add, aten.clamp, aten.round]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_clamp_div_mean_mul_round_sub_2.run(primals_5, buf3, 4, grid=grid(4), stream=stream0)
        del primals_5
        buf4 = reinterpret_tensor(buf2, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [add, output_1], Original ATen: [aten.add, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_sub_3.run(buf4, buf3, 256, grid=grid(256), stream=stream0)
        del buf3
    return (buf4, reinterpret_tensor(buf0, (64, 4), (4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
