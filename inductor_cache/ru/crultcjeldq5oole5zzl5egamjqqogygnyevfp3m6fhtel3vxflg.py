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


# kernel path: inductor_cache/it/citpzotlmr4ui3ccgi4l2mjxk4d4labc54ljmsuombleywieb7u6.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
triton_poi_fused_native_layer_norm_0 = async_compile.triton('triton_poi_fused_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = 4.0
    tmp8 = tmp6 / tmp7
    tmp9 = tmp0 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tmp1 - tmp8
    tmp12 = tmp11 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp3 - tmp8
    tmp15 = tmp14 * tmp14
    tmp16 = tmp13 + tmp15
    tmp17 = tmp5 - tmp8
    tmp18 = tmp17 * tmp17
    tmp19 = tmp16 + tmp18
    tmp20 = tmp19 / tmp7
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sm/csm3kauzljvoqykmhxhxprbjlalxtsjj5hfqqm3wplqhjfb6qipp.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_1, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_2), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_3), kwargs = {})
triton_poi_fused_native_layer_norm_1 = async_compile.triton('triton_poi_fused_native_layer_norm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 * tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tu/cturo3epnek3pofv2dqw2lvniazt3glfasgzysqlh5yizij4jxgm.py
# Topologically Sorted Source Nodes: [q, attn], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attn => clone
#   q => mul_2
# Graph fragment:
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%select, 1.0), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_2 = async_compile.triton('triton_poi_fused_clone_mul_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 12*x2 + 48*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x2 + 4*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jq/cjq5day5rbu2bqy5npupnl66pqri4wg2flml2vcebb7i5ezoahho.py
# Topologically Sorted Source Nodes: [attn], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (4 + y0 + 12*x2 + 48*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/op/copox25w4pf2us4f7tzlpcb4d467cnqap2ty3f5nxw67sz54zviq.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => amax, exp, sub_1
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_5, [-1], True), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_1,), kwargs = {})
triton_poi_fused__softmax_4 = async_compile.triton('triton_poi_fused__softmax_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/r7/cr7aeo43qxpygttm5oepwxkpnxd25y36cbh46hrqfbbfddrq5ato.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_5 = async_compile.triton('triton_poi_fused__softmax_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ju/cjucbifuc2zqitaidlpy435itf5ww7vhqevwnlzc2ycugg2gl5b4.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (8 + y0 + 12*x2 + 48*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/f2/cf2ifkanjv2mcrpm73rmugvzw63u52igxhe6xgyyyw65n2qnwxn7.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x_1 => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_9,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 16*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/qs/cqsksyyunsuz6kqwbppruagvjfvjhcrxubcavcrqskxix4xyzxfm.py
# Topologically Sorted Source Nodes: [x_1, x_3, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => var_mean_1
#   x_1 => add_2
#   x_3 => add_3
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %primals_6), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %add_2), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_3, [2]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_add_native_layer_norm_8 = async_compile.triton('triton_poi_fused_add_native_layer_norm_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 12, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp6 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (1))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp13 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr2 + (2))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp20 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr2 + (3))
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tmp5 = tmp0 + tmp4
    tmp10 = tmp7 + tmp9
    tmp11 = tmp6 + tmp10
    tmp12 = tmp5 + tmp11
    tmp17 = tmp14 + tmp16
    tmp18 = tmp13 + tmp17
    tmp19 = tmp12 + tmp18
    tmp24 = tmp21 + tmp23
    tmp25 = tmp20 + tmp24
    tmp26 = tmp19 + tmp25
    tmp27 = 4.0
    tmp28 = tmp26 / tmp27
    tmp29 = tmp5 - tmp28
    tmp30 = tmp29 * tmp29
    tmp31 = tmp11 - tmp28
    tmp32 = tmp31 * tmp31
    tmp33 = tmp30 + tmp32
    tmp34 = tmp18 - tmp28
    tmp35 = tmp34 * tmp34
    tmp36 = tmp33 + tmp35
    tmp37 = tmp25 - tmp28
    tmp38 = tmp37 * tmp37
    tmp39 = tmp36 + tmp38
    tmp40 = tmp39 / tmp27
    tl.store(out_ptr0 + (x0), tmp28, xmask)
    tl.store(out_ptr1 + (x0), tmp40, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/lf/clfofagqxmhookwu7nmux45akdiyrn64xvnbqtlwf5uizhc6zxob.py
# Topologically Sorted Source Nodes: [x_1, x_3, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_4, add_5, mul_3, mul_4, rsqrt_1, sub_2
#   x_1 => add_2
#   x_3 => add_3
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %primals_6), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %add_2), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_4,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_3, %getitem_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %primals_7), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %primals_8), kwargs = {})
triton_poi_fused_add_native_layer_norm_9 = async_compile.triton('triton_poi_fused_add_native_layer_norm_9', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp6 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uw/cuw3uvvlx2dj45x4j4pel5qefsydcbxxnuht24phkgfk2gwwdpev.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_5 => add_6, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_6), kwargs = {})
triton_poi_fused_gelu_10 = async_compile.triton('triton_poi_fused_gelu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kl/ckllrobsv4j3g5x6td6fmix7nhxhcmf5igj4rryykkxfkvmg27z2.py
# Topologically Sorted Source Nodes: [x_1, x_3, x_9], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_1 => add_2
#   x_3 => add_3
#   x_9 => add_7
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_11, %primals_6), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %add_2), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_3, %view_15), kwargs = {})
triton_poi_fused_add_11 = async_compile.triton('triton_poi_fused_add_11', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp6 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (12, 4), (4, 1))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (16, 4), (4, 1))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (4, 16), (16, 1))
    assert_size_stride(primals_12, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf1 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_0.run(primals_1, buf0, buf1, 16, grid=grid(16), stream=stream0)
        buf2 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_1.run(primals_1, buf0, buf1, primals_2, primals_3, buf2, 64, grid=grid(64), stream=stream0)
        del primals_2
        del primals_3
        buf3 = empty_strided_cuda((16, 12), (12, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (16, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 12), (1, 4), 0), out=buf3)
        buf4 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [q, attn], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_2.run(buf3, buf4, 16, 4, grid=grid(16, 4), stream=stream0)
        buf5 = empty_strided_cuda((4, 4, 1, 4), (16, 4, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf3, buf5, 16, 4, grid=grid(16, 4), stream=stream0)
        buf6 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf5, (16, 1, 4), (4, 0, 1), 0), out=buf6)
        buf7 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf6, buf7, 256, grid=grid(256), stream=stream0)
        buf8 = reinterpret_tensor(buf6, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf7, buf8, 256, grid=grid(256), stream=stream0)
        buf9 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf3, buf9, 16, 4, grid=grid(16, 4), stream=stream0)
        del buf3
        buf10 = empty_strided_cuda((16, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf9, (16, 4, 1), (4, 1, 0), 0), out=buf10)
        buf11 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_7.run(buf10, buf11, 16, 4, grid=grid(16, 4), stream=stream0)
        buf12 = reinterpret_tensor(buf10, (16, 4), (4, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf11, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 4), (1, 4), 0), out=buf12)
        buf13 = buf1; del buf1  # reuse
        buf14 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_3, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_8.run(primals_1, buf12, primals_6, buf13, buf14, 16, grid=grid(16), stream=stream0)
        buf15 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_3, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_9.run(primals_1, buf12, primals_6, buf13, buf14, primals_7, primals_8, buf15, 64, grid=grid(64), stream=stream0)
        del buf13
        del buf14
        del primals_8
        buf16 = reinterpret_tensor(buf7, (16, 16), (16, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf15, (16, 4), (4, 1), 0), reinterpret_tensor(primals_9, (4, 16), (1, 4), 0), alpha=1, beta=1, out=buf16)
        del primals_10
        buf17 = empty_strided_cuda((4, 4, 16), (64, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_10.run(buf16, buf17, 256, grid=grid(256), stream=stream0)
        buf18 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf17, (16, 16), (16, 1), 0), reinterpret_tensor(primals_11, (16, 4), (1, 16), 0), out=buf18)
        buf19 = reinterpret_tensor(buf18, (4, 4, 4), (16, 4, 1), 0); del buf18  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_3, x_9], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_11.run(buf19, primals_1, buf12, primals_6, primals_12, 64, grid=grid(64), stream=stream0)
        del primals_12
    return (buf19, primals_1, primals_6, primals_7, reinterpret_tensor(buf2, (16, 4), (4, 1), 0), buf8, reinterpret_tensor(buf11, (16, 4), (4, 1), 0), buf12, reinterpret_tensor(buf15, (16, 4), (4, 1), 0), buf16, reinterpret_tensor(buf17, (16, 16), (16, 1), 0), primals_11, primals_9, primals_5, reinterpret_tensor(buf9, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf4, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf5, (16, 4, 1), (4, 1, 4), 0), primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((12, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
