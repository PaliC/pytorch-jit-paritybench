# AOT ID: ['7_inference']
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


# kernel path: inductor_cache/lb/clbiq24do23ixoype4idtqqreepwh6aygbaxotuu3qbls2psiqwh.py
# Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   p_s => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%arg0_1, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%arg0_1, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
triton_poi_fused__softmax_0 = async_compile.triton('triton_poi_fused__softmax_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ry/cryk7ey74danxp6t4ch4lfscj5ml6xl53kb2hmfcfcotgflxrgeo.py
# Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   p_s => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=4] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_1 = async_compile.triton('triton_poi_fused__softmax_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cw/ccwnvjwfrn5v4vw336dirapphf7abpdwh4gut7mi4mnj4lwv2pyg.py
# Topologically Sorted Source Nodes: [neg, truediv, K, sum_1, x], Original ATen: [aten.neg, aten.div, aten.exp, aten.sum]
# Source node to ATen node mapping:
#   K => exp_2
#   neg => neg
#   sum_1 => sum_3
#   truediv => div_2
#   x => div_3
# Graph fragment:
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%_cdist_forward,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%neg, 0.1), kwargs = {})
#   %exp_2 : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%div_2,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_2, [1], True), kwargs = {})
#   %div_3 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_2, %sum_3), kwargs = {})
triton_poi_fused_div_exp_neg_sum_2 = async_compile.triton('triton_poi_fused_div_exp_neg_sum_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_exp_neg_sum_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_exp_neg_sum_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp5 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (16 + x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr0 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr0 + (48 + x0), xmask, eviction_policy='evict_last')
    tmp1 = -tmp0
    tmp2 = 10.0
    tmp3 = tmp1 * tmp2
    tmp4 = tl_math.exp(tmp3)
    tmp6 = -tmp5
    tmp7 = tmp6 * tmp2
    tmp8 = tl_math.exp(tmp7)
    tmp10 = -tmp9
    tmp11 = tmp10 * tmp2
    tmp12 = tl_math.exp(tmp11)
    tmp13 = tmp8 + tmp12
    tmp15 = -tmp14
    tmp16 = tmp15 * tmp2
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp13 + tmp17
    tmp20 = -tmp19
    tmp21 = tmp20 * tmp2
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp18 + tmp22
    tmp24 = tmp4 / tmp23
    tl.store(out_ptr0 + (x2), tmp24, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/do/cdo3qwjatsq2p7yeep5tyhweiwaiblkuz3mfejf4hxmr5goybc2v.py
# Topologically Sorted Source Nodes: [sum_2, x_1, sum_3, x_2], Original ATen: [aten.sum, aten.div]
# Source node to ATen node mapping:
#   sum_2 => sum_4
#   sum_3 => sum_5
#   x_1 => div_4
#   x_2 => div_5
# Graph fragment:
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_3, [0], True), kwargs = {})
#   %div_4 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_3, %sum_4), kwargs = {})
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_4, [1], True), kwargs = {})
#   %div_5 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_4, %sum_5), kwargs = {})
triton_poi_fused_div_sum_3 = async_compile.triton('triton_poi_fused_div_sum_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sum_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sum_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (16 + x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (32 + x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (48 + x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 / tmp0
    tmp3 = tmp2 / tmp2
    tmp5 = tmp4 / tmp4
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 / tmp7
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10 / tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = tmp1 / tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/g2/cg2xkp7dj4jmjerzslaxrqpq7ob4lsoeks6rghab2jx3hkodm76f.py
# Topologically Sorted Source Nodes: [sum_38, x_37, sum_39, x_38, sum_40, x_39, mul, sum_41, emd_loss, sum_79, x_77, sum_80, x_78, sum_81, x_79, mul_1, sum_82, emd_loss_1, sum_120, x_117, sum_121, x_118, sum_122, x_119, mul_2, sum_123, emd_loss_2, sum_161, x_157, sum_162, x_158, sum_163, x_159, mul_3, sum_164, emd_loss_3, mul_4], Original ATen: [aten.sum, aten.div, aten.mul, aten.add]
# Source node to ATen node mapping:
#   emd_loss => add
#   emd_loss_1 => add_1
#   emd_loss_2 => add_2
#   emd_loss_3 => add_3
#   mul => mul
#   mul_1 => mul_1
#   mul_2 => mul_2
#   mul_3 => mul_3
#   mul_4 => mul_4
#   sum_120 => sum_122
#   sum_121 => sum_123
#   sum_122 => sum_124
#   sum_123 => sum_125
#   sum_161 => sum_163
#   sum_162 => sum_164
#   sum_163 => sum_165
#   sum_164 => sum_166
#   sum_38 => sum_40
#   sum_39 => sum_41
#   sum_40 => sum_42
#   sum_41 => sum_43
#   sum_79 => sum_81
#   sum_80 => sum_82
#   sum_81 => sum_83
#   sum_82 => sum_84
#   x_117 => div_122
#   x_118 => div_123
#   x_119 => div_124
#   x_157 => div_163
#   x_158 => div_164
#   x_159 => div_165
#   x_37 => div_40
#   x_38 => div_41
#   x_39 => div_42
#   x_77 => div_81
#   x_78 => div_82
#   x_79 => div_83
# Graph fragment:
#   %sum_40 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_39, [0], True), kwargs = {})
#   %div_40 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_39, %sum_40), kwargs = {})
#   %sum_41 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_40, [1], True), kwargs = {})
#   %div_41 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_40, %sum_41), kwargs = {})
#   %sum_42 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_41, [0], True), kwargs = {})
#   %div_42 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_41, %sum_42), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_42, %_cdist_forward), kwargs = {})
#   %sum_43 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%sum_43, 0.0), kwargs = {})
#   %sum_81 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_80, [0], True), kwargs = {})
#   %div_81 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_80, %sum_81), kwargs = {})
#   %sum_82 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_81, [1], True), kwargs = {})
#   %div_82 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_81, %sum_82), kwargs = {})
#   %sum_83 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_82, [0], True), kwargs = {})
#   %div_83 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_82, %sum_83), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_83, %_cdist_forward_1), kwargs = {})
#   %sum_84 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_1,), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %sum_84), kwargs = {})
#   %sum_122 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_121, [0], True), kwargs = {})
#   %div_122 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_121, %sum_122), kwargs = {})
#   %sum_123 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_122, [1], True), kwargs = {})
#   %div_123 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_122, %sum_123), kwargs = {})
#   %sum_124 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_123, [0], True), kwargs = {})
#   %div_124 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_123, %sum_124), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_124, %_cdist_forward_2), kwargs = {})
#   %sum_125 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_2,), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1, %sum_125), kwargs = {})
#   %sum_163 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_162, [0], True), kwargs = {})
#   %div_163 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_162, %sum_163), kwargs = {})
#   %sum_164 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_163, [1], True), kwargs = {})
#   %div_164 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_163, %sum_164), kwargs = {})
#   %sum_165 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%div_164, [0], True), kwargs = {})
#   %div_165 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_164, %sum_165), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div_165, %_cdist_forward_3), kwargs = {})
#   %sum_166 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%mul_3,), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %sum_166), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_3, 0.001), kwargs = {})
triton_per_fused_add_div_mul_sum_4 = async_compile.triton('triton_per_fused_add_div_mul_sum_4', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 10), 'tt.equal_to': (9,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_sum_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 24, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_div_mul_sum_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    r0 = (rindex % 16)
    tmp0 = tl.load(in_ptr0 + (r2), None)
    tmp2 = tl.load(in_ptr0 + (r0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (16 + r0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (32 + r0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (48 + r0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (r2), None)
    tmp20 = tl.load(in_ptr2 + (r2), None)
    tmp22 = tl.load(in_ptr2 + (r0), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr2 + (16 + r0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (32 + r0), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr2 + (48 + r0), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr3 + (r2), None)
    tmp40 = tl.load(in_ptr4 + (r2), None)
    tmp42 = tl.load(in_ptr4 + (r0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr4 + (16 + r0), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (32 + r0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr4 + (48 + r0), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (r2), None)
    tmp60 = tl.load(in_ptr6 + (r2), None)
    tmp62 = tl.load(in_ptr6 + (r0), None, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr6 + (16 + r0), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr6 + (32 + r0), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr6 + (48 + r0), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr7 + (r2), None)
    tmp1 = tmp0 / tmp0
    tmp3 = tmp2 / tmp2
    tmp5 = tmp4 / tmp4
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7 / tmp7
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10 / tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = tmp1 / tmp12
    tmp14 = tmp13 / tmp13
    tmp16 = tmp14 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None]
    tmp21 = tmp20 / tmp20
    tmp23 = tmp22 / tmp22
    tmp25 = tmp24 / tmp24
    tmp26 = tmp23 + tmp25
    tmp28 = tmp27 / tmp27
    tmp29 = tmp26 + tmp28
    tmp31 = tmp30 / tmp30
    tmp32 = tmp29 + tmp31
    tmp33 = tmp21 / tmp32
    tmp34 = tmp33 / tmp33
    tmp36 = tmp34 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
    tmp39 = tl.sum(tmp37, 1)[:, None]
    tmp41 = tmp40 / tmp40
    tmp43 = tmp42 / tmp42
    tmp45 = tmp44 / tmp44
    tmp46 = tmp43 + tmp45
    tmp48 = tmp47 / tmp47
    tmp49 = tmp46 + tmp48
    tmp51 = tmp50 / tmp50
    tmp52 = tmp49 + tmp51
    tmp53 = tmp41 / tmp52
    tmp54 = tmp53 / tmp53
    tmp56 = tmp54 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp59 = tl.sum(tmp57, 1)[:, None]
    tmp61 = tmp60 / tmp60
    tmp63 = tmp62 / tmp62
    tmp65 = tmp64 / tmp64
    tmp66 = tmp63 + tmp65
    tmp68 = tmp67 / tmp67
    tmp69 = tmp66 + tmp68
    tmp71 = tmp70 / tmp70
    tmp72 = tmp69 + tmp71
    tmp73 = tmp61 / tmp72
    tmp74 = tmp73 / tmp73
    tmp76 = tmp74 * tmp75
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp79 = tl.sum(tmp77, 1)[:, None]
    tmp80 = 0.0
    tmp81 = tmp19 + tmp80
    tmp82 = tmp81 + tmp39
    tmp83 = tmp82 + tmp59
    tmp84 = tmp83 + tmp79
    tmp85 = 0.001
    tmp86 = tmp84 * tmp85
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp86, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg0_1, buf0, 256, grid=grid(256), stream=stream0)
        del arg0_1
        buf2 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_t], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_0.run(arg1_1, buf2, 256, grid=grid(256), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [p_s], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf0, buf1, 256, grid=grid(256), stream=stream0)
        buf3 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [p_t], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf2, buf3, 256, grid=grid(256), stream=stream0)
        del buf2
        # Topologically Sorted Source Nodes: [Wxy], Original ATen: [aten._cdist_forward]
        buf4 = torch.ops.aten._cdist_forward.default(reinterpret_tensor(buf1, (1, 4, 4, 4), (64, 16, 4, 1), 0), reinterpret_tensor(buf3, (1, 4, 4, 4), (64, 16, 4, 1), 0), 1.0, None)
        # Topologically Sorted Source Nodes: [Wxy_1], Original ATen: [aten._cdist_forward]
        buf27 = torch.ops.aten._cdist_forward.default(reinterpret_tensor(buf1, (1, 4, 4, 4), (64, 16, 4, 1), 64), reinterpret_tensor(buf3, (1, 4, 4, 4), (64, 16, 4, 1), 64), 1.0, None)
        # Topologically Sorted Source Nodes: [Wxy_2], Original ATen: [aten._cdist_forward]
        buf50 = torch.ops.aten._cdist_forward.default(reinterpret_tensor(buf1, (1, 4, 4, 4), (64, 16, 4, 1), 128), reinterpret_tensor(buf3, (1, 4, 4, 4), (64, 16, 4, 1), 128), 1.0, None)
        # Topologically Sorted Source Nodes: [Wxy_3], Original ATen: [aten._cdist_forward]
        buf73 = torch.ops.aten._cdist_forward.default(reinterpret_tensor(buf1, (1, 4, 4, 4), (64, 16, 4, 1), 192), reinterpret_tensor(buf3, (1, 4, 4, 4), (64, 16, 4, 1), 192), 1.0, None)
        del buf1
        del buf3
        buf5 = buf4
        del buf4
        buf28 = buf27
        del buf27
        buf51 = buf50
        del buf50
        buf74 = buf73
        del buf73
        buf6 = empty_strided_cuda((1, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg, truediv, K, sum_1, x], Original ATen: [aten.neg, aten.div, aten.exp, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_exp_neg_sum_2.run(buf5, buf6, 64, grid=grid(64), stream=stream0)
        buf29 = empty_strided_cuda((1, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg_1, truediv_41, K_1, sum_42, x_40], Original ATen: [aten.neg, aten.div, aten.exp, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_exp_neg_sum_2.run(buf28, buf29, 64, grid=grid(64), stream=stream0)
        buf52 = empty_strided_cuda((1, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg_2, truediv_82, K_2, sum_83, x_80], Original ATen: [aten.neg, aten.div, aten.exp, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_exp_neg_sum_2.run(buf51, buf52, 64, grid=grid(64), stream=stream0)
        buf75 = empty_strided_cuda((1, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [neg_3, truediv_123, K_3, sum_124, x_120], Original ATen: [aten.neg, aten.div, aten.exp, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_exp_neg_sum_2.run(buf74, buf75, 64, grid=grid(64), stream=stream0)
        buf7 = empty_strided_cuda((1, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sum_2, x_1, sum_3, x_2], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf6, buf7, 64, grid=grid(64), stream=stream0)
        buf30 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [sum_43, x_41, sum_44, x_42], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf29, buf30, 64, grid=grid(64), stream=stream0)
        buf53 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [sum_84, x_81, sum_85, x_82], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf52, buf53, 64, grid=grid(64), stream=stream0)
        buf76 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [sum_125, x_121, sum_126, x_122], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf75, buf76, 64, grid=grid(64), stream=stream0)
        buf8 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [sum_4, x_3, sum_5, x_4], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf7, buf8, 64, grid=grid(64), stream=stream0)
        buf31 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [sum_45, x_43, sum_46, x_44], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf30, buf31, 64, grid=grid(64), stream=stream0)
        buf54 = buf30; del buf30  # reuse
        # Topologically Sorted Source Nodes: [sum_86, x_83, sum_87, x_84], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf53, buf54, 64, grid=grid(64), stream=stream0)
        buf77 = buf53; del buf53  # reuse
        # Topologically Sorted Source Nodes: [sum_127, x_123, sum_128, x_124], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf76, buf77, 64, grid=grid(64), stream=stream0)
        buf9 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [sum_6, x_5, sum_7, x_6], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf8, buf9, 64, grid=grid(64), stream=stream0)
        buf32 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [sum_47, x_45, sum_48, x_46], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf31, buf32, 64, grid=grid(64), stream=stream0)
        buf55 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [sum_88, x_85, sum_89, x_86], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf54, buf55, 64, grid=grid(64), stream=stream0)
        buf78 = buf54; del buf54  # reuse
        # Topologically Sorted Source Nodes: [sum_129, x_125, sum_130, x_126], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf77, buf78, 64, grid=grid(64), stream=stream0)
        buf10 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [sum_8, x_7, sum_9, x_8], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf9, buf10, 64, grid=grid(64), stream=stream0)
        buf33 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [sum_49, x_47, sum_50, x_48], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf32, buf33, 64, grid=grid(64), stream=stream0)
        buf56 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [sum_90, x_87, sum_91, x_88], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf55, buf56, 64, grid=grid(64), stream=stream0)
        buf79 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [sum_131, x_127, sum_132, x_128], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf78, buf79, 64, grid=grid(64), stream=stream0)
        buf11 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [sum_10, x_9, sum_11, x_10], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf10, buf11, 64, grid=grid(64), stream=stream0)
        buf34 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [sum_51, x_49, sum_52, x_50], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf33, buf34, 64, grid=grid(64), stream=stream0)
        buf57 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [sum_92, x_89, sum_93, x_90], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf56, buf57, 64, grid=grid(64), stream=stream0)
        buf80 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [sum_133, x_129, sum_134, x_130], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf79, buf80, 64, grid=grid(64), stream=stream0)
        buf12 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [sum_12, x_11, sum_13, x_12], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf11, buf12, 64, grid=grid(64), stream=stream0)
        buf35 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [sum_53, x_51, sum_54, x_52], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf34, buf35, 64, grid=grid(64), stream=stream0)
        buf58 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [sum_94, x_91, sum_95, x_92], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf57, buf58, 64, grid=grid(64), stream=stream0)
        buf81 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [sum_135, x_131, sum_136, x_132], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf80, buf81, 64, grid=grid(64), stream=stream0)
        buf13 = buf80; del buf80  # reuse
        # Topologically Sorted Source Nodes: [sum_14, x_13, sum_15, x_14], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf12, buf13, 64, grid=grid(64), stream=stream0)
        buf36 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [sum_55, x_53, sum_56, x_54], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf35, buf36, 64, grid=grid(64), stream=stream0)
        buf59 = buf35; del buf35  # reuse
        # Topologically Sorted Source Nodes: [sum_96, x_93, sum_97, x_94], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf58, buf59, 64, grid=grid(64), stream=stream0)
        buf82 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [sum_137, x_133, sum_138, x_134], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf81, buf82, 64, grid=grid(64), stream=stream0)
        buf14 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [sum_16, x_15, sum_17, x_16], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf13, buf14, 64, grid=grid(64), stream=stream0)
        buf37 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [sum_57, x_55, sum_58, x_56], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf36, buf37, 64, grid=grid(64), stream=stream0)
        buf60 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [sum_98, x_95, sum_99, x_96], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf59, buf60, 64, grid=grid(64), stream=stream0)
        buf83 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [sum_139, x_135, sum_140, x_136], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf82, buf83, 64, grid=grid(64), stream=stream0)
        buf15 = buf82; del buf82  # reuse
        # Topologically Sorted Source Nodes: [sum_18, x_17, sum_19, x_18], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf14, buf15, 64, grid=grid(64), stream=stream0)
        buf38 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [sum_59, x_57, sum_60, x_58], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf37, buf38, 64, grid=grid(64), stream=stream0)
        buf61 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [sum_100, x_97, sum_101, x_98], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf60, buf61, 64, grid=grid(64), stream=stream0)
        buf84 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [sum_141, x_137, sum_142, x_138], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf83, buf84, 64, grid=grid(64), stream=stream0)
        buf16 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [sum_20, x_19, sum_21, x_20], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf15, buf16, 64, grid=grid(64), stream=stream0)
        buf39 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [sum_61, x_59, sum_62, x_60], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf38, buf39, 64, grid=grid(64), stream=stream0)
        buf62 = buf38; del buf38  # reuse
        # Topologically Sorted Source Nodes: [sum_102, x_99, sum_103, x_100], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf61, buf62, 64, grid=grid(64), stream=stream0)
        buf85 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [sum_143, x_139, sum_144, x_140], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf84, buf85, 64, grid=grid(64), stream=stream0)
        buf17 = buf84; del buf84  # reuse
        # Topologically Sorted Source Nodes: [sum_22, x_21, sum_23, x_22], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf16, buf17, 64, grid=grid(64), stream=stream0)
        buf40 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [sum_63, x_61, sum_64, x_62], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf39, buf40, 64, grid=grid(64), stream=stream0)
        buf63 = buf39; del buf39  # reuse
        # Topologically Sorted Source Nodes: [sum_104, x_101, sum_105, x_102], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf62, buf63, 64, grid=grid(64), stream=stream0)
        buf86 = buf62; del buf62  # reuse
        # Topologically Sorted Source Nodes: [sum_145, x_141, sum_146, x_142], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf85, buf86, 64, grid=grid(64), stream=stream0)
        buf18 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [sum_24, x_23, sum_25, x_24], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf17, buf18, 64, grid=grid(64), stream=stream0)
        buf41 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [sum_65, x_63, sum_66, x_64], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf40, buf41, 64, grid=grid(64), stream=stream0)
        buf64 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [sum_106, x_103, sum_107, x_104], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf63, buf64, 64, grid=grid(64), stream=stream0)
        buf87 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [sum_147, x_143, sum_148, x_144], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf86, buf87, 64, grid=grid(64), stream=stream0)
        buf19 = buf86; del buf86  # reuse
        # Topologically Sorted Source Nodes: [sum_26, x_25, sum_27, x_26], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf18, buf19, 64, grid=grid(64), stream=stream0)
        buf42 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [sum_67, x_65, sum_68, x_66], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf41, buf42, 64, grid=grid(64), stream=stream0)
        buf65 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [sum_108, x_105, sum_109, x_106], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf64, buf65, 64, grid=grid(64), stream=stream0)
        buf88 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [sum_149, x_145, sum_150, x_146], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf87, buf88, 64, grid=grid(64), stream=stream0)
        buf20 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [sum_28, x_27, sum_29, x_28], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf19, buf20, 64, grid=grid(64), stream=stream0)
        buf43 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [sum_69, x_67, sum_70, x_68], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf42, buf43, 64, grid=grid(64), stream=stream0)
        buf66 = buf42; del buf42  # reuse
        # Topologically Sorted Source Nodes: [sum_110, x_107, sum_111, x_108], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf65, buf66, 64, grid=grid(64), stream=stream0)
        buf89 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [sum_151, x_147, sum_152, x_148], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf88, buf89, 64, grid=grid(64), stream=stream0)
        buf21 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [sum_30, x_29, sum_31, x_30], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf20, buf21, 64, grid=grid(64), stream=stream0)
        buf44 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [sum_71, x_69, sum_72, x_70], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf43, buf44, 64, grid=grid(64), stream=stream0)
        buf67 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [sum_112, x_109, sum_113, x_110], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf66, buf67, 64, grid=grid(64), stream=stream0)
        buf90 = buf66; del buf66  # reuse
        # Topologically Sorted Source Nodes: [sum_153, x_149, sum_154, x_150], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf89, buf90, 64, grid=grid(64), stream=stream0)
        buf22 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [sum_32, x_31, sum_33, x_32], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf21, buf22, 64, grid=grid(64), stream=stream0)
        buf45 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [sum_73, x_71, sum_74, x_72], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf44, buf45, 64, grid=grid(64), stream=stream0)
        buf68 = buf44; del buf44  # reuse
        # Topologically Sorted Source Nodes: [sum_114, x_111, sum_115, x_112], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf67, buf68, 64, grid=grid(64), stream=stream0)
        buf91 = buf67; del buf67  # reuse
        # Topologically Sorted Source Nodes: [sum_155, x_151, sum_156, x_152], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf90, buf91, 64, grid=grid(64), stream=stream0)
        buf23 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [sum_34, x_33, sum_35, x_34], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf22, buf23, 64, grid=grid(64), stream=stream0)
        buf46 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [sum_75, x_73, sum_76, x_74], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf45, buf46, 64, grid=grid(64), stream=stream0)
        buf69 = buf45; del buf45  # reuse
        # Topologically Sorted Source Nodes: [sum_116, x_113, sum_117, x_114], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf68, buf69, 64, grid=grid(64), stream=stream0)
        buf92 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [sum_157, x_153, sum_158, x_154], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf91, buf92, 64, grid=grid(64), stream=stream0)
        buf24 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [sum_36, x_35, sum_37, x_36], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf23, buf24, 64, grid=grid(64), stream=stream0)
        buf47 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [sum_77, x_75, sum_78, x_76], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf46, buf47, 64, grid=grid(64), stream=stream0)
        buf70 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [sum_118, x_115, sum_119, x_116], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf69, buf70, 64, grid=grid(64), stream=stream0)
        buf93 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [sum_159, x_155, sum_160, x_156], Original ATen: [aten.sum, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sum_3.run(buf92, buf93, 64, grid=grid(64), stream=stream0)
        del buf92
        buf26 = empty_strided_cuda((), (), torch.float32)
        buf96 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [sum_38, x_37, sum_39, x_38, sum_40, x_39, mul, sum_41, emd_loss, sum_79, x_77, sum_80, x_78, sum_81, x_79, mul_1, sum_82, emd_loss_1, sum_120, x_117, sum_121, x_118, sum_122, x_119, mul_2, sum_123, emd_loss_2, sum_161, x_157, sum_162, x_158, sum_163, x_159, mul_3, sum_164, emd_loss_3, mul_4], Original ATen: [aten.sum, aten.div, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_mul_sum_4.run(buf96, buf24, buf5, buf47, buf28, buf70, buf51, buf93, buf74, 1, 64, grid=grid(1), stream=stream0)
        del buf24
        del buf28
        del buf47
        del buf5
        del buf51
        del buf70
        del buf74
        del buf93
    return (buf96, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
