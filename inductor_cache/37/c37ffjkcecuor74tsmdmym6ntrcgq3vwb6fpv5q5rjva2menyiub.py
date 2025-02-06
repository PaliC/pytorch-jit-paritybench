# AOT ID: ['64_forward']
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


# kernel path: inductor_cache/nw/cnwmug5e2ecu5spyvto4zcwdymnn446p5btfyxhu5pwpwgypc3qc.py
# Topologically Sorted Source Nodes: [x1, x2, out], Original ATen: [aten.avg_pool2d, aten.max_pool2d_with_indices, aten.clone]
# Source node to ATen node mapping:
#   out => clone
#   x1 => avg_pool2d
#   x2 => _low_memory_max_pool2d_with_offsets
# Graph fragment:
#   %avg_pool2d : [num_users=1] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%primals_1, [3, 5], [2, 1], [1, 2]), kwargs = {})
#   %_low_memory_max_pool2d_with_offsets : [num_users=1] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%primals_1, [3, 5], [2, 1], [1, 2], [1, 1], False), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_avg_pool2d_clone_max_pool2d_with_indices_0 = async_compile.triton('triton_poi_fused_avg_pool2d_clone_max_pool2d_with_indices_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr2': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_clone_max_pool2d_with_indices_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 30, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_clone_max_pool2d_with_indices_0(in_ptr0, out_ptr2, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 8
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex // 4
    x1 = (xindex % 4)
    y0 = yindex
    x5 = xindex
    y3 = (yindex % 4)
    y4 = yindex // 4
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-2) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6) + x1 + 8*x2 + 16*y0), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = (-1) + x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5) + x1 + 8*x2 + 16*y0), tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-4) + x1 + 8*x2 + 16*y0), tmp23 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = 1 + x1
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp5 & tmp29
    tmp31 = tl.load(in_ptr0 + ((-3) + x1 + 8*x2 + 16*y0), tmp30 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = 2 + x1
    tmp34 = tmp33 >= tmp1
    tmp35 = tmp33 < tmp3
    tmp36 = tmp34 & tmp35
    tmp37 = tmp5 & tmp36
    tmp38 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp37 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp38 + tmp32
    tmp40 = 2*x2
    tmp41 = tmp40 >= tmp1
    tmp42 = tmp40 < tmp3
    tmp43 = tmp41 & tmp42
    tmp44 = tmp43 & tmp9
    tmp45 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp44 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp45 + tmp39
    tmp47 = tmp43 & tmp15
    tmp48 = tl.load(in_ptr0 + ((-1) + x1 + 8*x2 + 16*y0), tmp47 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp48 + tmp46
    tmp50 = tmp43 & tmp22
    tmp51 = tl.load(in_ptr0 + (x1 + 8*x2 + 16*y0), tmp50 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp52 = tmp51 + tmp49
    tmp53 = tmp43 & tmp29
    tmp54 = tl.load(in_ptr0 + (1 + x1 + 8*x2 + 16*y0), tmp53 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp54 + tmp52
    tmp56 = tmp43 & tmp36
    tmp57 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp56 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp58 = tmp57 + tmp55
    tmp59 = 1 + 2*x2
    tmp60 = tmp59 >= tmp1
    tmp61 = tmp59 < tmp3
    tmp62 = tmp60 & tmp61
    tmp63 = tmp62 & tmp9
    tmp64 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp63 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp65 = tmp64 + tmp58
    tmp66 = tmp62 & tmp15
    tmp67 = tl.load(in_ptr0 + (3 + x1 + 8*x2 + 16*y0), tmp66 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp67 + tmp65
    tmp69 = tmp62 & tmp22
    tmp70 = tl.load(in_ptr0 + (4 + x1 + 8*x2 + 16*y0), tmp69 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp71 = tmp70 + tmp68
    tmp72 = tmp62 & tmp29
    tmp73 = tl.load(in_ptr0 + (5 + x1 + 8*x2 + 16*y0), tmp72 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp74 = tmp73 + tmp71
    tmp75 = tmp62 & tmp36
    tmp76 = tl.load(in_ptr0 + (6 + x1 + 8*x2 + 16*y0), tmp75 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp77 = tmp76 + tmp74
    tmp78 = 2 + ((-1)*x1) + ((-4)*x2) + 2*((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5))) + ((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5)))*((6) * ((6) <= (3 + x1)) + (3 + x1) * ((3 + x1) < (6))) + ((-1)*x1*((5) * ((5) <= (2 + 2*x2)) + (2 + 2*x2) * ((2 + 2*x2) < (5)))) + ((-2)*x2*((6) * ((6) <= (3 + x1)) + (3 + x1) * ((3 + x1) < (6)))) + 2*x1*x2 + ((6) * ((6) <= (3 + x1)) + (3 + x1) * ((3 + x1) < (6)))
    tmp79 = tmp77 / tmp78
    tmp80 = tl.load(in_ptr0 + ((-6) + x1 + 8*x2 + 16*y0), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp81 = tl.load(in_ptr0 + ((-5) + x1 + 8*x2 + 16*y0), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp82 = triton_helpers.maximum(tmp81, tmp80)
    tmp83 = tl.load(in_ptr0 + ((-4) + x1 + 8*x2 + 16*y0), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp84 = triton_helpers.maximum(tmp83, tmp82)
    tmp85 = tl.load(in_ptr0 + ((-3) + x1 + 8*x2 + 16*y0), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp86 = triton_helpers.maximum(tmp85, tmp84)
    tmp87 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp37 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp88 = triton_helpers.maximum(tmp87, tmp86)
    tmp89 = tl.load(in_ptr0 + ((-2) + x1 + 8*x2 + 16*y0), tmp44 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp90 = triton_helpers.maximum(tmp89, tmp88)
    tmp91 = tl.load(in_ptr0 + ((-1) + x1 + 8*x2 + 16*y0), tmp47 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp92 = triton_helpers.maximum(tmp91, tmp90)
    tmp93 = tl.load(in_ptr0 + (x1 + 8*x2 + 16*y0), tmp50 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp94 = triton_helpers.maximum(tmp93, tmp92)
    tmp95 = tl.load(in_ptr0 + (1 + x1 + 8*x2 + 16*y0), tmp53 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp96 = triton_helpers.maximum(tmp95, tmp94)
    tmp97 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp56 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp98 = triton_helpers.maximum(tmp97, tmp96)
    tmp99 = tl.load(in_ptr0 + (2 + x1 + 8*x2 + 16*y0), tmp63 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp100 = triton_helpers.maximum(tmp99, tmp98)
    tmp101 = tl.load(in_ptr0 + (3 + x1 + 8*x2 + 16*y0), tmp66 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp102 = triton_helpers.maximum(tmp101, tmp100)
    tmp103 = tl.load(in_ptr0 + (4 + x1 + 8*x2 + 16*y0), tmp69 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp104 = triton_helpers.maximum(tmp103, tmp102)
    tmp105 = tl.load(in_ptr0 + (5 + x1 + 8*x2 + 16*y0), tmp72 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp106 = triton_helpers.maximum(tmp105, tmp104)
    tmp107 = tl.load(in_ptr0 + (6 + x1 + 8*x2 + 16*y0), tmp75 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp108 = triton_helpers.maximum(tmp107, tmp106)
    tmp109 = tmp79 + tmp108
    tmp110 = 0.5
    tmp111 = tmp109 * tmp110
    tl.store(out_ptr2 + (y3 + 4*x5 + 32*y4), tmp111, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/bz/cbzivr6dxtzeaaioojars5n2bmmvhhnnlehvnj45f4tzt2n6lgsr.py
# Topologically Sorted Source Nodes: [out, out_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   out => add_1
#   out_1 => var_mean
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %primals_3), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_add_native_layer_norm_1 = async_compile.triton('triton_poi_fused_add_native_layer_norm_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK])
    tmp4 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr1 + (1))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp9 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (2))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp14 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr1 + (3))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp3 = tmp0 + tmp2
    tmp7 = tmp4 + tmp6
    tmp8 = tmp3 + tmp7
    tmp12 = tmp9 + tmp11
    tmp13 = tmp8 + tmp12
    tmp17 = tmp14 + tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = 4.0
    tmp20 = tmp18 / tmp19
    tmp21 = tmp3 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tmp7 - tmp20
    tmp24 = tmp23 * tmp23
    tmp25 = tmp22 + tmp24
    tmp26 = tmp12 - tmp20
    tmp27 = tmp26 * tmp26
    tmp28 = tmp25 + tmp27
    tmp29 = tmp17 - tmp20
    tmp30 = tmp29 * tmp29
    tmp31 = tmp28 + tmp30
    tmp32 = tmp31 / tmp19
    tl.store(out_ptr0 + (x0), tmp20, xmask)
    tl.store(out_ptr1 + (x0), tmp32, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ca/ccan2sicqkullh2ukgu6d4e6272rrlc6uwjujtt7ghg7dok25z5b.py
# Topologically Sorted Source Nodes: [out, out_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   out => add_1
#   out_1 => add_2, add_3, mul_1, mul_2, rsqrt, sub
# Graph fragment:
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_2, %primals_3), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_3), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_4), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_5), kwargs = {})
triton_poi_fused_add_native_layer_norm_2 = async_compile.triton('triton_poi_fused_add_native_layer_norm_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 128}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp4 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf2 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x1, x2, out], Original ATen: [aten.avg_pool2d, aten.max_pool2d_with_indices, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_clone_max_pool2d_with_indices_0.run(primals_1, buf2, 16, 8, grid=grid(16, 8), stream=stream0)
        del primals_1
        buf3 = empty_strided_cuda((32, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (32, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf3)
        del primals_2
        buf4 = empty_strided_cuda((4, 8, 1), (8, 1, 32), torch.float32)
        buf5 = empty_strided_cuda((4, 8, 1), (8, 1, 32), torch.float32)
        # Topologically Sorted Source Nodes: [out, out_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_1.run(buf3, primals_3, buf4, buf5, 32, grid=grid(32), stream=stream0)
        buf6 = empty_strided_cuda((4, 8, 4), (32, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out, out_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_2.run(buf3, primals_3, buf4, buf5, primals_4, primals_5, buf6, 128, grid=grid(128), stream=stream0)
        del buf4
        del buf5
        del primals_5
    return (buf6, primals_3, primals_4, reinterpret_tensor(buf2, (32, 4), (4, 1), 0), buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
