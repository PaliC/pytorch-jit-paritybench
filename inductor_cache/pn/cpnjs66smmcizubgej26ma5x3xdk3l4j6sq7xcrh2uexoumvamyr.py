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


# kernel path: inductor_cache/yk/cykars5hvde4dapa43zxv5cue236drt34o2yzsa3wa2o46swestr.py
# Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm => add, add_1, mul, mul_1, rsqrt, sub, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%primals_3, [2]), kwargs = {correction: 0, keepdim: True})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_3, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_1), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_2), kwargs = {})
triton_per_fused_native_layer_norm_0 = async_compile.triton('triton_per_fused_native_layer_norm_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 3, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 384.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp0 - tmp10
    tmp23 = tmp22 * tmp21
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp21, None)
    tl.store(out_ptr1 + (r1 + 384*x0), tmp27, rmask)
    tl.store(out_ptr0 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ic/cicomybz64miec7ondmwthlzuiogsk72aqxbbwkoditftdtqc6gq.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 4)
    x2 = ((xindex // 256) % 6)
    x3 = xindex // 1536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 1152*x1 + 4608*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l4/cl4ddfepell3qtrx2skbeekv7ncxtqbhjxc7ukkj2ibm5popso4i.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone_1
# Graph fragment:
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 384)
    y1 = yindex // 384
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (384 + y0 + 1152*x2 + 4608*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ng/cngbwhheq6mgdoipzziwclsyhgebzx5afrgm2tk5anmdpcuerxwi.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => exp
# Graph fragment:
#   %mul_tensor_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_5, 1), kwargs = {})
#   %amax_default_5 : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%mul_tensor_10, [-1], True), kwargs = {})
#   %sub_tensor_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_tensor_10, %amax_default_5), kwargs = {})
#   %mul_tensor_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_tensor_5, 0.125), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%mul_tensor_11,), kwargs = {})
triton_poi_fused__softmax_3 = async_compile.triton('triton_poi_fused__softmax_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp3 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = 1.0
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 * tmp1
    tmp6 = tmp5 * tmp1
    tmp7 = triton_helpers.maximum(tmp4, tmp6)
    tmp9 = tmp8 * tmp1
    tmp10 = triton_helpers.maximum(tmp7, tmp9)
    tmp12 = tmp11 * tmp1
    tmp13 = triton_helpers.maximum(tmp10, tmp12)
    tmp14 = tmp2 - tmp13
    tmp15 = 0.125
    tmp16 = tmp14 * tmp15
    tmp17 = tl_math.exp(tmp16)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zq/czqe3utpirdddzm6nj33v75utgzlvamfx7krkxk5fcslvzicwxsq.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_4 = async_compile.triton('triton_poi_fused__softmax_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 384
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


# kernel path: inductor_cache/tk/ctknig4bexlwkp3psh3txvp2styqujqe3q3x3jkwj64ncxnavsp2.py
# Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 4)
    x2 = ((xindex // 256) % 6)
    x3 = xindex // 1536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (768 + x0 + 64*x2 + 1152*x1 + 4608*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/c3/cc3lxjgzdgnx4leeoowivey3l3fvog2cqdr5zurkcyuyiqppx2rq.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   x => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 6)
    x2 = ((xindex // 384) % 4)
    x3 = xindex // 1536
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64*x2 + 256*x1 + 1536*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/d2/cd26b4rnla2lqihkcifeibbu7fgwhgli3b3v6mxrngllhhrvmro7.py
# Topologically Sorted Source Nodes: [x_3, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_3, add_4, mul_3, mul_4, rsqrt_1, sub_2, var_mean_1
#   x_3 => add_2
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %view_11), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_3), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_3, %primals_7), kwargs = {})
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_4, %primals_8), kwargs = {})
triton_per_fused_add_native_layer_norm_7 = async_compile.triton('triton_per_fused_add_native_layer_norm_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 4, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 384*x0), rmask, other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp10 = tl.full([1], 384, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 / tmp11
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = 384.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tmp24 = tmp2 - tmp12
    tmp25 = tmp24 * tmp23
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, None)
    tl.store(out_ptr1 + (r1 + 384*x0), tmp29, rmask)
    tl.store(out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/rq/crqed7ngcopz6fexgjgdjm5gqu5idchzenptdmgc3o7oh2bf5amf.py
# Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   x_5 => add_5, erf, mul_5, mul_6, mul_7
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.5), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_13, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_6,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_5, %add_5), kwargs = {})
triton_poi_fused_gelu_8 = async_compile.triton('triton_poi_fused_gelu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_gelu_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp0 * tmp3
    tmp5 = libdevice.erf(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp2 * tmp7
    tl.store(out_ptr0 + (x0), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/ei/ceih53ckwqer4jl3uprl2rwajeykp7lgcvqo2npzrrpipmvfjqxu.py
# Topologically Sorted Source Nodes: [x_3, x_9, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_2 => add_7, add_8, mul_8, mul_9, rsqrt_2, sub_3, var_mean_2
#   x_3 => add_2
#   x_9 => add_6
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %view_11), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_15), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt_2 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_5), kwargs = {})
#   %mul_8 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_2), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %primals_13), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_9, %primals_14), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_2, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 6, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 384*x0), rmask, other=0.0)
    tmp3 = tl.load(in_ptr2 + (r1 + 384*x0), rmask, other=0.0)
    tmp4 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 384, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 384.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = libdevice.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = 0.0026041666666666665
    tmp35 = tmp28 * tmp34
    tl.store(out_ptr2 + (r1 + 384*x0), tmp29, rmask)
    tl.store(out_ptr3 + (r1 + 384*x0), tmp33, rmask)
    tl.store(out_ptr4 + (x0), tmp35, None)
''', device_str='cuda')


# kernel path: inductor_cache/7f/c7fwutbok7izdl5tfao4dmx2znlhkd2ilcwbsn2mrjzmb2iggj3f.py
# Topologically Sorted Source Nodes: [x_3, x_9, x_13, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_3 => add_10, add_11, mul_11, mul_12, rsqrt_3, sub_5, var_mean_3
#   x_13 => add_9
#   x_3 => add_2
#   x_9 => add_6
# Graph fragment:
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_3, %view_11), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_15), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_27), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_7), kwargs = {})
#   %mul_11 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_3), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_11, %primals_18), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %primals_19), kwargs = {})
#   %div_15 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 8, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 384*x0), rmask, other=0.0)
    tmp3 = tl.load(in_out_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp4 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr3 + (r1 + 384*x0), rmask, other=0.0)
    tmp8 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 384, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 384.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = 0.0026041666666666665
    tmp39 = tmp32 * tmp38
    tl.store(in_out_ptr0 + (r1 + 384*x0), tmp10, rmask)
    tl.store(out_ptr2 + (r1 + 384*x0), tmp33, rmask)
    tl.store(out_ptr3 + (r1 + 384*x0), tmp37, rmask)
    tl.store(out_ptr4 + (x0), tmp39, None)
''', device_str='cuda')


# kernel path: inductor_cache/d7/cd7qxjr7aq64ukzmlzlktlspf5n2cnr3t45owvn5j2eu5nxyni3t.py
# Topologically Sorted Source Nodes: [x_19, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_4 => add_14, add_15, mul_16, mul_17, rsqrt_4, sub_6, var_mean_4
#   x_19 => add_13
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_31), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_13, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_13, %getitem_9), kwargs = {})
#   %mul_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %rsqrt_4), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %primals_24), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %primals_25), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + 384*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0026041666666666665
    tmp33 = tmp26 * tmp32
    tl.store(out_ptr2 + (r1 + 384*x0), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + 384*x0), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp33, None)
''', device_str='cuda')


# kernel path: inductor_cache/mj/cmjedu33wrfhcgxqsbuxso22l3bwrhyngpyx3jmmi7nhz5enqr2f.py
# Topologically Sorted Source Nodes: [x_19, x_23, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_5 => add_17, add_18, mul_19, mul_20, rsqrt_5, sub_8, var_mean_5
#   x_19 => add_13
#   x_23 => add_16
# Graph fragment:
#   %add_13 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_31), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_13, %view_43), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_16, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_10, 1e-05), kwargs = {})
#   %rsqrt_5 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_17,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_16, %getitem_11), kwargs = {})
#   %mul_19 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_5), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %primals_29), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %primals_30), kwargs = {})
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_5, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 7, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r1 + 384*x0), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = tl.full([1], 384, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = triton_helpers.promote_to_tensor(tl.sum(tmp23, 0))
    tmp25 = tmp8 - tmp18
    tmp26 = 384.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = libdevice.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = 0.0026041666666666665
    tmp37 = tmp30 * tmp36
    tl.store(in_out_ptr0 + (r1 + 384*x0), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + 384*x0), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + 384*x0), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp37, None)
''', device_str='cuda')


# kernel path: inductor_cache/be/cbe56fsr7rd4oeotfuvjzurvptakcfbmx4jjrzns254x3kkohfci.py
# Topologically Sorted Source Nodes: [x_59, layer_norm_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   layer_norm_12 => add_42, add_43, mul_48, mul_49, rsqrt_12, sub_18, var_mean_12
#   x_59 => add_41
# Graph fragment:
#   %add_41 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_37, %view_95), kwargs = {})
#   %var_mean_12 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_41, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_42 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_24, 1e-05), kwargs = {})
#   %rsqrt_12 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_42,), kwargs = {})
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_41, %getitem_25), kwargs = {})
#   %mul_48 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %rsqrt_12), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_48, %primals_68), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_49, %primals_69), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_12, 384), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 512},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': True, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 16
    XBLOCK: tl.constexpr = 1
    rnumel = 384
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = tl.full([RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r1 + 384*x0), rmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = triton_helpers.promote_to_tensor(tl.sum(tmp10, 0))
    tmp12 = tl.full([1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = triton_helpers.promote_to_tensor(tl.sum(tmp19, 0))
    tmp21 = tmp4 - tmp14
    tmp22 = 384.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0026041666666666665
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r1 + 384*x0), tmp27, rmask)
    tl.store(out_ptr2 + (r1 + 384*x0), tmp31, rmask)
    tl.store(out_ptr3 + (x0), tmp33, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71 = args
    args.clear()
    assert_size_stride(primals_1, (384, ), (1, ))
    assert_size_stride(primals_2, (384, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 384), (1536, 384, 1))
    assert_size_stride(primals_4, (1152, 384), (384, 1))
    assert_size_stride(primals_5, (384, 384), (384, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (384, ), (1, ))
    assert_size_stride(primals_8, (384, ), (1, ))
    assert_size_stride(primals_9, (1536, 384), (384, 1))
    assert_size_stride(primals_10, (1536, ), (1, ))
    assert_size_stride(primals_11, (384, 1536), (1536, 1))
    assert_size_stride(primals_12, (384, ), (1, ))
    assert_size_stride(primals_13, (384, ), (1, ))
    assert_size_stride(primals_14, (384, ), (1, ))
    assert_size_stride(primals_15, (1152, 384), (384, 1))
    assert_size_stride(primals_16, (384, 384), (384, 1))
    assert_size_stride(primals_17, (384, ), (1, ))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_20, (1536, 384), (384, 1))
    assert_size_stride(primals_21, (1536, ), (1, ))
    assert_size_stride(primals_22, (384, 1536), (1536, 1))
    assert_size_stride(primals_23, (384, ), (1, ))
    assert_size_stride(primals_24, (384, ), (1, ))
    assert_size_stride(primals_25, (384, ), (1, ))
    assert_size_stride(primals_26, (1152, 384), (384, 1))
    assert_size_stride(primals_27, (384, 384), (384, 1))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (384, ), (1, ))
    assert_size_stride(primals_30, (384, ), (1, ))
    assert_size_stride(primals_31, (1536, 384), (384, 1))
    assert_size_stride(primals_32, (1536, ), (1, ))
    assert_size_stride(primals_33, (384, 1536), (1536, 1))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (384, ), (1, ))
    assert_size_stride(primals_36, (384, ), (1, ))
    assert_size_stride(primals_37, (1152, 384), (384, 1))
    assert_size_stride(primals_38, (384, 384), (384, 1))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (1536, 384), (384, 1))
    assert_size_stride(primals_43, (1536, ), (1, ))
    assert_size_stride(primals_44, (384, 1536), (1536, 1))
    assert_size_stride(primals_45, (384, ), (1, ))
    assert_size_stride(primals_46, (384, ), (1, ))
    assert_size_stride(primals_47, (384, ), (1, ))
    assert_size_stride(primals_48, (1152, 384), (384, 1))
    assert_size_stride(primals_49, (384, 384), (384, 1))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (384, ), (1, ))
    assert_size_stride(primals_53, (1536, 384), (384, 1))
    assert_size_stride(primals_54, (1536, ), (1, ))
    assert_size_stride(primals_55, (384, 1536), (1536, 1))
    assert_size_stride(primals_56, (384, ), (1, ))
    assert_size_stride(primals_57, (384, ), (1, ))
    assert_size_stride(primals_58, (384, ), (1, ))
    assert_size_stride(primals_59, (1152, 384), (384, 1))
    assert_size_stride(primals_60, (384, 384), (384, 1))
    assert_size_stride(primals_61, (384, ), (1, ))
    assert_size_stride(primals_62, (384, ), (1, ))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (1536, 384), (384, 1))
    assert_size_stride(primals_65, (1536, ), (1, ))
    assert_size_stride(primals_66, (384, 1536), (1536, 1))
    assert_size_stride(primals_67, (384, ), (1, ))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (256, 384), (384, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        buf1 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf3 = reinterpret_tensor(buf1, (4, 4, 1), (4, 1, 1), 0); del buf1  # reuse
        buf4 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [layer_norm], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_native_layer_norm_0.run(buf3, primals_3, primals_1, primals_2, buf0, buf4, 16, 384, grid=grid(16), stream=stream0)
        del primals_1
        del primals_2
        buf5 = empty_strided_cuda((16, 1152), (1152, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (16, 384), (384, 1), 0), reinterpret_tensor(primals_4, (384, 1152), (1, 384), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf5, buf6, 6144, grid=grid(6144), stream=stream0)
        buf7 = empty_strided_cuda((4, 6, 64, 4), (1536, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf5, buf7, 1536, 4, grid=grid(1536, 4), stream=stream0)
        buf8 = empty_strided_cuda((24, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (24, 4, 64), (256, 64, 1), 0), reinterpret_tensor(buf7, (24, 64, 4), (256, 4, 1), 0), out=buf8)
        buf9 = empty_strided_cuda((4, 6, 4, 4), (96, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf8, buf9, 384, grid=grid(384), stream=stream0)
        buf10 = reinterpret_tensor(buf8, (4, 6, 4, 4), (96, 16, 4, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf9, buf10, 384, grid=grid(384), stream=stream0)
        buf11 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf5, buf11, 6144, grid=grid(6144), stream=stream0)
        buf12 = empty_strided_cuda((24, 4, 64), (256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (24, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf11, (24, 4, 64), (256, 64, 1), 0), out=buf12)
        buf13 = empty_strided_cuda((4, 4, 6, 64), (1536, 384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf12, buf13, 6144, grid=grid(6144), stream=stream0)
        buf14 = reinterpret_tensor(buf12, (16, 384), (384, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf13, (16, 384), (384, 1), 0), reinterpret_tensor(primals_5, (384, 384), (1, 384), 0), alpha=1, beta=1, out=buf14)
        del primals_6
        buf15 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        buf16 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf18 = reinterpret_tensor(buf16, (4, 4, 1), (4, 1, 1), 0); del buf16  # reuse
        buf19 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, layer_norm_1], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_7.run(buf18, primals_3, buf14, primals_7, primals_8, buf15, buf19, 16, 384, grid=grid(16), stream=stream0)
        del primals_8
        buf20 = empty_strided_cuda((16, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf19, (16, 384), (384, 1), 0), reinterpret_tensor(primals_9, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf20)
        del primals_10
        buf21 = empty_strided_cuda((4, 4, 1536), (6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf20, buf21, 24576, grid=grid(24576), stream=stream0)
        buf22 = empty_strided_cuda((16, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf21, (16, 1536), (1536, 1), 0), reinterpret_tensor(primals_11, (1536, 384), (1, 1536), 0), out=buf22)
        buf26 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf27 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf159 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_9, layer_norm_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_9.run(primals_3, buf14, buf22, primals_12, primals_13, primals_14, buf26, buf27, buf159, 16, 384, grid=grid(16), stream=stream0)
        del primals_14
        buf28 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf27, (16, 384), (384, 1), 0), reinterpret_tensor(primals_15, (384, 1152), (1, 384), 0), out=buf28)
        buf29 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf28, buf29, 6144, grid=grid(6144), stream=stream0)
        buf30 = empty_strided_cuda((4, 6, 64, 4), (1536, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf28, buf30, 1536, 4, grid=grid(1536, 4), stream=stream0)
        buf31 = reinterpret_tensor(buf9, (24, 4, 4), (16, 4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (24, 4, 64), (256, 64, 1), 0), reinterpret_tensor(buf30, (24, 64, 4), (256, 4, 1), 0), out=buf31)
        buf32 = empty_strided_cuda((4, 6, 4, 4), (96, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf31, buf32, 384, grid=grid(384), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (4, 6, 4, 4), (96, 16, 4, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [attn_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf32, buf33, 384, grid=grid(384), stream=stream0)
        buf34 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf28, buf34, 6144, grid=grid(6144), stream=stream0)
        buf35 = empty_strided_cuda((24, 4, 64), (256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (24, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf34, (24, 4, 64), (256, 64, 1), 0), out=buf35)
        buf36 = empty_strided_cuda((4, 4, 6, 64), (1536, 384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf35, buf36, 6144, grid=grid(6144), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (16, 384), (384, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf36, (16, 384), (384, 1), 0), reinterpret_tensor(primals_16, (384, 384), (1, 384), 0), out=buf37)
        buf38 = reinterpret_tensor(buf22, (4, 4, 384), (1536, 384, 1), 0); del buf22  # reuse
        buf42 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf43 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf158 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3, x_9, x_13, layer_norm_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf38, primals_3, buf14, primals_12, buf37, primals_17, primals_18, primals_19, buf42, buf43, buf158, 16, 384, grid=grid(16), stream=stream0)
        del primals_12
        del primals_17
        del primals_19
        buf44 = empty_strided_cuda((16, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_21, reinterpret_tensor(buf43, (16, 384), (384, 1), 0), reinterpret_tensor(primals_20, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf44)
        del primals_21
        buf45 = empty_strided_cuda((4, 4, 1536), (6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf44, buf45, 24576, grid=grid(24576), stream=stream0)
        buf46 = buf37; del buf37  # reuse
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf45, (16, 1536), (1536, 1), 0), reinterpret_tensor(primals_22, (1536, 384), (1, 1536), 0), out=buf46)
        buf50 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf51 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf157 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, layer_norm_4], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf38, buf46, primals_23, primals_24, primals_25, buf50, buf51, buf157, 16, 384, grid=grid(16), stream=stream0)
        del primals_25
        buf52 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf51, (16, 384), (384, 1), 0), reinterpret_tensor(primals_26, (384, 1152), (1, 384), 0), out=buf52)
        buf53 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf52, buf53, 6144, grid=grid(6144), stream=stream0)
        buf54 = empty_strided_cuda((4, 6, 64, 4), (1536, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf52, buf54, 1536, 4, grid=grid(1536, 4), stream=stream0)
        buf55 = reinterpret_tensor(buf32, (24, 4, 4), (16, 4, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf53, (24, 4, 64), (256, 64, 1), 0), reinterpret_tensor(buf54, (24, 64, 4), (256, 4, 1), 0), out=buf55)
        buf56 = empty_strided_cuda((4, 6, 4, 4), (96, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_7], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf55, buf56, 384, grid=grid(384), stream=stream0)
        buf57 = reinterpret_tensor(buf55, (4, 6, 4, 4), (96, 16, 4, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [attn_7], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf56, buf57, 384, grid=grid(384), stream=stream0)
        buf58 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf52, buf58, 6144, grid=grid(6144), stream=stream0)
        buf59 = empty_strided_cuda((24, 4, 64), (256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf57, (24, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf58, (24, 4, 64), (256, 64, 1), 0), out=buf59)
        buf60 = empty_strided_cuda((4, 4, 6, 64), (1536, 384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf59, buf60, 6144, grid=grid(6144), stream=stream0)
        buf61 = reinterpret_tensor(buf59, (16, 384), (384, 1), 0); del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf60, (16, 384), (384, 1), 0), reinterpret_tensor(primals_27, (384, 384), (1, 384), 0), out=buf61)
        buf62 = buf38; del buf38  # reuse
        buf66 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf67 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf156 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_23, layer_norm_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf62, buf46, primals_23, buf61, primals_28, primals_29, primals_30, buf66, buf67, buf156, 16, 384, grid=grid(16), stream=stream0)
        del primals_23
        del primals_28
        del primals_30
        buf68 = empty_strided_cuda((16, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_32, reinterpret_tensor(buf67, (16, 384), (384, 1), 0), reinterpret_tensor(primals_31, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf68)
        del primals_32
        buf69 = empty_strided_cuda((4, 4, 1536), (6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf68, buf69, 24576, grid=grid(24576), stream=stream0)
        buf70 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf69, (16, 1536), (1536, 1), 0), reinterpret_tensor(primals_33, (1536, 384), (1, 1536), 0), out=buf70)
        buf74 = reinterpret_tensor(buf46, (4, 4, 384), (1536, 384, 1), 0); del buf46  # reuse
        buf75 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf155 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_29, layer_norm_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf62, buf70, primals_34, primals_35, primals_36, buf74, buf75, buf155, 16, 384, grid=grid(16), stream=stream0)
        del primals_36
        buf76 = buf52; del buf52  # reuse
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (16, 384), (384, 1), 0), reinterpret_tensor(primals_37, (384, 1152), (1, 384), 0), out=buf76)
        buf77 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf76, buf77, 6144, grid=grid(6144), stream=stream0)
        buf78 = empty_strided_cuda((4, 6, 64, 4), (1536, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf76, buf78, 1536, 4, grid=grid(1536, 4), stream=stream0)
        buf79 = reinterpret_tensor(buf56, (24, 4, 4), (16, 4, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf77, (24, 4, 64), (256, 64, 1), 0), reinterpret_tensor(buf78, (24, 64, 4), (256, 4, 1), 0), out=buf79)
        buf80 = empty_strided_cuda((4, 6, 4, 4), (96, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_10], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf79, buf80, 384, grid=grid(384), stream=stream0)
        buf81 = reinterpret_tensor(buf79, (4, 6, 4, 4), (96, 16, 4, 1), 0); del buf79  # reuse
        # Topologically Sorted Source Nodes: [attn_10], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf80, buf81, 384, grid=grid(384), stream=stream0)
        buf82 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf76, buf82, 6144, grid=grid(6144), stream=stream0)
        buf83 = empty_strided_cuda((24, 4, 64), (256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf81, (24, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf82, (24, 4, 64), (256, 64, 1), 0), out=buf83)
        buf84 = empty_strided_cuda((4, 4, 6, 64), (1536, 384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf83, buf84, 6144, grid=grid(6144), stream=stream0)
        buf85 = reinterpret_tensor(buf83, (16, 384), (384, 1), 0); del buf83  # reuse
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf84, (16, 384), (384, 1), 0), reinterpret_tensor(primals_38, (384, 384), (1, 384), 0), out=buf85)
        buf86 = buf62; del buf62  # reuse
        buf90 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf91 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf154 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_29, x_33, layer_norm_7], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf86, buf70, primals_34, buf85, primals_39, primals_40, primals_41, buf90, buf91, buf154, 16, 384, grid=grid(16), stream=stream0)
        del primals_34
        del primals_39
        del primals_41
        buf92 = empty_strided_cuda((16, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_43, reinterpret_tensor(buf91, (16, 384), (384, 1), 0), reinterpret_tensor(primals_42, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf92)
        del primals_43
        buf93 = empty_strided_cuda((4, 4, 1536), (6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf92, buf93, 24576, grid=grid(24576), stream=stream0)
        buf94 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf93, (16, 1536), (1536, 1), 0), reinterpret_tensor(primals_44, (1536, 384), (1, 1536), 0), out=buf94)
        buf98 = reinterpret_tensor(buf70, (4, 4, 384), (1536, 384, 1), 0); del buf70  # reuse
        buf99 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf153 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_39, layer_norm_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf86, buf94, primals_45, primals_46, primals_47, buf98, buf99, buf153, 16, 384, grid=grid(16), stream=stream0)
        del primals_47
        buf100 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (16, 384), (384, 1), 0), reinterpret_tensor(primals_48, (384, 1152), (1, 384), 0), out=buf100)
        buf101 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf100, buf101, 6144, grid=grid(6144), stream=stream0)
        buf102 = empty_strided_cuda((4, 6, 64, 4), (1536, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf100, buf102, 1536, 4, grid=grid(1536, 4), stream=stream0)
        buf103 = reinterpret_tensor(buf80, (24, 4, 4), (16, 4, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf101, (24, 4, 64), (256, 64, 1), 0), reinterpret_tensor(buf102, (24, 64, 4), (256, 4, 1), 0), out=buf103)
        buf104 = empty_strided_cuda((4, 6, 4, 4), (96, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_13], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf103, buf104, 384, grid=grid(384), stream=stream0)
        buf105 = reinterpret_tensor(buf103, (4, 6, 4, 4), (96, 16, 4, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [attn_13], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf104, buf105, 384, grid=grid(384), stream=stream0)
        buf106 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf100, buf106, 6144, grid=grid(6144), stream=stream0)
        buf107 = empty_strided_cuda((24, 4, 64), (256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf105, (24, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf106, (24, 4, 64), (256, 64, 1), 0), out=buf107)
        buf108 = empty_strided_cuda((4, 4, 6, 64), (1536, 384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf107, buf108, 6144, grid=grid(6144), stream=stream0)
        buf109 = reinterpret_tensor(buf107, (16, 384), (384, 1), 0); del buf107  # reuse
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf108, (16, 384), (384, 1), 0), reinterpret_tensor(primals_49, (384, 384), (1, 384), 0), out=buf109)
        buf110 = buf86; del buf86  # reuse
        buf114 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf115 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf152 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_39, x_43, layer_norm_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf110, buf94, primals_45, buf109, primals_50, primals_51, primals_52, buf114, buf115, buf152, 16, 384, grid=grid(16), stream=stream0)
        del primals_45
        del primals_50
        del primals_52
        buf116 = empty_strided_cuda((16, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_54, reinterpret_tensor(buf115, (16, 384), (384, 1), 0), reinterpret_tensor(primals_53, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf116)
        del primals_54
        buf117 = empty_strided_cuda((4, 4, 1536), (6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf116, buf117, 24576, grid=grid(24576), stream=stream0)
        buf118 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf117, (16, 1536), (1536, 1), 0), reinterpret_tensor(primals_55, (1536, 384), (1, 1536), 0), out=buf118)
        buf122 = reinterpret_tensor(buf109, (4, 4, 384), (1536, 384, 1), 0); del buf109  # reuse
        buf123 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf151 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_49, layer_norm_10], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_11.run(buf110, buf118, primals_56, primals_57, primals_58, buf122, buf123, buf151, 16, 384, grid=grid(16), stream=stream0)
        del primals_58
        buf124 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (16, 384), (384, 1), 0), reinterpret_tensor(primals_59, (384, 1152), (1, 384), 0), out=buf124)
        buf125 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf124, buf125, 6144, grid=grid(6144), stream=stream0)
        buf126 = empty_strided_cuda((4, 6, 64, 4), (1536, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf124, buf126, 1536, 4, grid=grid(1536, 4), stream=stream0)
        buf127 = reinterpret_tensor(buf104, (24, 4, 4), (16, 4, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (24, 4, 64), (256, 64, 1), 0), reinterpret_tensor(buf126, (24, 64, 4), (256, 4, 1), 0), out=buf127)
        buf128 = empty_strided_cuda((4, 6, 4, 4), (96, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_16], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_3.run(buf127, buf128, 384, grid=grid(384), stream=stream0)
        buf129 = reinterpret_tensor(buf127, (4, 6, 4, 4), (96, 16, 4, 1), 0); del buf127  # reuse
        # Topologically Sorted Source Nodes: [attn_16], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf128, buf129, 384, grid=grid(384), stream=stream0)
        del buf128
        buf130 = empty_strided_cuda((4, 6, 4, 64), (1536, 256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf124, buf130, 6144, grid=grid(6144), stream=stream0)
        del buf124
        buf131 = empty_strided_cuda((24, 4, 64), (256, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf129, (24, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf130, (24, 4, 64), (256, 64, 1), 0), out=buf131)
        buf132 = empty_strided_cuda((4, 4, 6, 64), (1536, 384, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf131, buf132, 6144, grid=grid(6144), stream=stream0)
        buf133 = reinterpret_tensor(buf131, (16, 384), (384, 1), 0); del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf132, (16, 384), (384, 1), 0), reinterpret_tensor(primals_60, (384, 384), (1, 384), 0), out=buf133)
        buf134 = buf110; del buf110  # reuse
        buf138 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf139 = empty_strided_cuda((4, 4, 384), (1536, 384, 1), torch.float32)
        buf150 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_49, x_53, layer_norm_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_12.run(buf134, buf118, primals_56, buf133, primals_61, primals_62, primals_63, buf138, buf139, buf150, 16, 384, grid=grid(16), stream=stream0)
        del primals_56
        del primals_61
        del primals_63
        buf140 = empty_strided_cuda((16, 1536), (1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_65, reinterpret_tensor(buf139, (16, 384), (384, 1), 0), reinterpret_tensor(primals_64, (384, 1536), (1, 384), 0), alpha=1, beta=1, out=buf140)
        del primals_65
        buf141 = empty_strided_cuda((4, 4, 1536), (6144, 1536, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf140, buf141, 24576, grid=grid(24576), stream=stream0)
        buf142 = buf133; del buf133  # reuse
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf141, (16, 1536), (1536, 1), 0), reinterpret_tensor(primals_66, (1536, 384), (1, 1536), 0), out=buf142)
        buf146 = buf134; del buf134  # reuse
        buf147 = reinterpret_tensor(buf118, (4, 4, 384), (1536, 384, 1), 0); del buf118  # reuse
        buf149 = empty_strided_cuda((4, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_59, layer_norm_12], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_13.run(buf146, buf142, primals_67, primals_68, primals_69, buf147, buf149, 16, 384, grid=grid(16), stream=stream0)
        del buf142
        del primals_67
        del primals_69
        buf148 = empty_strided_cuda((16, 256), (256, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_71, reinterpret_tensor(buf147, (16, 384), (384, 1), 0), reinterpret_tensor(primals_70, (384, 256), (1, 384), 0), alpha=1, beta=1, out=buf148)
        del primals_71
    return (reinterpret_tensor(buf148, (4, 4, 256), (1024, 256, 1), 0), primals_3, primals_7, primals_13, primals_18, primals_24, primals_29, primals_35, primals_40, primals_46, primals_51, primals_57, primals_62, primals_68, buf0, buf3, reinterpret_tensor(buf4, (16, 384), (384, 1), 0), buf10, reinterpret_tensor(buf13, (16, 384), (384, 1), 0), buf14, buf15, buf18, reinterpret_tensor(buf19, (16, 384), (384, 1), 0), buf20, reinterpret_tensor(buf21, (16, 1536), (1536, 1), 0), buf26, reinterpret_tensor(buf27, (16, 384), (384, 1), 0), buf33, reinterpret_tensor(buf36, (16, 384), (384, 1), 0), buf42, reinterpret_tensor(buf43, (16, 384), (384, 1), 0), buf44, reinterpret_tensor(buf45, (16, 1536), (1536, 1), 0), buf50, reinterpret_tensor(buf51, (16, 384), (384, 1), 0), buf57, reinterpret_tensor(buf60, (16, 384), (384, 1), 0), buf66, reinterpret_tensor(buf67, (16, 384), (384, 1), 0), buf68, reinterpret_tensor(buf69, (16, 1536), (1536, 1), 0), buf74, reinterpret_tensor(buf75, (16, 384), (384, 1), 0), buf81, reinterpret_tensor(buf84, (16, 384), (384, 1), 0), buf90, reinterpret_tensor(buf91, (16, 384), (384, 1), 0), buf92, reinterpret_tensor(buf93, (16, 1536), (1536, 1), 0), buf98, reinterpret_tensor(buf99, (16, 384), (384, 1), 0), buf105, reinterpret_tensor(buf108, (16, 384), (384, 1), 0), buf114, reinterpret_tensor(buf115, (16, 384), (384, 1), 0), buf116, reinterpret_tensor(buf117, (16, 1536), (1536, 1), 0), buf122, reinterpret_tensor(buf123, (16, 384), (384, 1), 0), buf129, reinterpret_tensor(buf132, (16, 384), (384, 1), 0), buf138, reinterpret_tensor(buf139, (16, 384), (384, 1), 0), buf140, reinterpret_tensor(buf141, (16, 1536), (1536, 1), 0), buf146, reinterpret_tensor(buf147, (16, 384), (384, 1), 0), primals_70, buf149, primals_66, primals_64, buf150, primals_60, reinterpret_tensor(buf130, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf125, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf126, (24, 4, 64), (256, 1, 4), 0), primals_59, buf151, primals_55, primals_53, buf152, primals_49, reinterpret_tensor(buf106, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf101, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf102, (24, 4, 64), (256, 1, 4), 0), primals_48, buf153, primals_44, primals_42, buf154, primals_38, reinterpret_tensor(buf82, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf77, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf78, (24, 4, 64), (256, 1, 4), 0), primals_37, buf155, primals_33, primals_31, buf156, primals_27, reinterpret_tensor(buf58, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf53, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf54, (24, 4, 64), (256, 1, 4), 0), primals_26, buf157, primals_22, primals_20, buf158, primals_16, reinterpret_tensor(buf34, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf29, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf30, (24, 4, 64), (256, 1, 4), 0), primals_15, buf159, primals_11, primals_9, primals_5, reinterpret_tensor(buf11, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf6, (24, 64, 4), (256, 1, 64), 0), reinterpret_tensor(buf7, (24, 4, 64), (256, 1, 4), 0), primals_4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 384), (1536, 384, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1152, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1536, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, 1536), (1536, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
