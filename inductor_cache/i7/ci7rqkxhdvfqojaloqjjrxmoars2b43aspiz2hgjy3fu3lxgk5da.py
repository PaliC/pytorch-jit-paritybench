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


# kernel path: inductor_cache/5f/c5ftqdtf2yn7sfse2kw4dyr7q2gqyfptb6zrstik6pky5nolwdnk.py
# Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul => mul_scalar
# Graph fragment:
#   %mul_scalar : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_4, 0.7071067811865476), kwargs = {})
triton_poi_fused_mul_0 = async_compile.triton('triton_poi_fused_mul_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mq/cmqgdrujltwhxj23znt7uylb3yyk6a7ncumexhtutetg5f4x4vba.py
# Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
# Source node to ATen node mapping:
#   mul_1 => mul_scalar_1
# Graph fragment:
#   %mul_scalar_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_default, 0.7071067811865476), kwargs = {})
triton_poi_fused_mul_1 = async_compile.triton('triton_poi_fused_mul_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
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
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.7071067811865476
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/aq/caqmiiysszdztukbxpdadqcm4rg3wrb25tz5oxkaqfpcxwrmwqoe.py
# Topologically Sorted Source Nodes: [amax, sub, exp], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   amax => amax_default
#   exp => exp_default
#   sub => sub_tensor
# Graph fragment:
#   %amax_default : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_default_2, [-1], True), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_default_2, %amax_default), kwargs = {})
#   %exp_default : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
triton_poi_fused__safe_softmax_2 = async_compile.triton('triton_poi_fused__safe_softmax_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/4p/c4ph4efdt2x2rcd5fshygpsghu7ot3bszmxuqgwt43owdxv5aw5d.py
# Topologically Sorted Source Nodes: [sum_1, div, eq, logical_not, any_1, logical_not_1, full, where], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   any_1 => any_dim
#   div => div_tensor
#   eq => eq_scalar
#   full => full_default
#   logical_not => logical_not_default
#   logical_not_1 => logical_not_default_1
#   sum_1 => sum_dim_int_list
#   where => where_self
# Graph fragment:
#   %sum_dim_int_list : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp_default, [-1], True), kwargs = {})
#   %div_tensor : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %sum_dim_int_list), kwargs = {})
#   %eq_scalar : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_default_2, -inf), kwargs = {})
#   %logical_not_default : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_scalar,), kwargs = {})
#   %any_dim : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_default, -1, True), kwargs = {})
#   %logical_not_default_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_dim,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 1, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_self : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_default_1, %full_default, %div_tensor), kwargs = {})
triton_poi_fused__safe_softmax_3 = async_compile.triton('triton_poi_fused__safe_softmax_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr1 + (x2), xmask)
    tmp26 = tl.load(in_ptr1 + (4*x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr1 + (1 + 4*x1), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (2 + 4*x1), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr1 + (3 + 4*x1), xmask, eviction_policy='evict_last')
    tmp1 = float("-inf")
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2 == 0
    tmp4 = tmp3.to(tl.int64)
    tmp5 = (tmp4 != 0)
    tmp7 = tmp6 == tmp1
    tmp8 = tmp7 == 0
    tmp9 = tmp8.to(tl.int64)
    tmp10 = (tmp9 != 0)
    tmp11 = tmp5 | tmp10
    tmp13 = tmp12 == tmp1
    tmp14 = tmp13 == 0
    tmp15 = tmp14.to(tl.int64)
    tmp16 = (tmp15 != 0)
    tmp17 = tmp11 | tmp16
    tmp19 = tmp18 == tmp1
    tmp20 = tmp19 == 0
    tmp21 = tmp20.to(tl.int64)
    tmp22 = (tmp21 != 0)
    tmp23 = tmp17 | tmp22
    tmp24 = tmp23 == 0
    tmp28 = tmp26 + tmp27
    tmp30 = tmp28 + tmp29
    tmp32 = tmp30 + tmp31
    tmp33 = tmp25 / tmp32
    tmp34 = 0.0
    tmp35 = tl.where(tmp24, tmp34, tmp33)
    tl.store(out_ptr0 + (x2), tmp35, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oy/coy3peearnubhaaj4me554trl3wsl6g7wtssw3m527onselu6igb.py
# Topologically Sorted Source Nodes: [permute_4], Original ATen: [aten.transpose]
# Source node to ATen node mapping:
#   permute_4 => permute_default_4
# Graph fragment:
#   %permute_default_4 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_default_1, [0, 2, 1]), kwargs = {})
triton_poi_fused_transpose_4 = async_compile.triton('triton_poi_fused_transpose_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_transpose_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_transpose_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_7, (4, 4), (4, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, 4), (4, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [keys], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_1, (4, 4), (1, 4), 0), out=buf0)
        del primals_1
        buf1 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [queries], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_6, (64, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf1)
        del primals_4
        buf2 = empty_strided_cuda((64, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [values], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_8, reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_7, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf2)
        del primals_7
        del primals_8
        buf3 = reinterpret_tensor(buf1, (4, 4, 1, 4, 4), (64, 16, 16, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_0.run(buf3, primals_5, 256, grid=grid(256), stream=stream0)
        del primals_5
        buf4 = empty_strided_cuda((4, 4, 1, 4, 4), (64, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_1.run(buf0, primals_2, buf4, 64, 4, grid=grid(64, 4), stream=stream0)
        del primals_2
        buf5 = reinterpret_tensor(buf0, (16, 4, 4), (16, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [bmm], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf4, (16, 4, 4), (16, 4, 1), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 4, 1, 4, 4), (64, 16, 256, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [amax, sub, exp], Original ATen: [aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_2.run(buf5, buf6, 256, grid=grid(256), stream=stream0)
        buf7 = empty_strided_cuda((4, 4, 1, 4, 4), (64, 16, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sum_1, div, eq, logical_not, any_1, logical_not_1, full, where], Original ATen: [aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_3.run(buf5, buf6, buf7, 256, grid=grid(256), stream=stream0)
        buf8 = reinterpret_tensor(buf6, (16, 4, 4), (16, 4, 1), 0); del buf6  # reuse
        # Topologically Sorted Source Nodes: [bmm_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf7, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf2, (16, 4, 4), (16, 4, 1), 0), out=buf8)
        buf9 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [permute_4], Original ATen: [aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_transpose_4.run(buf4, buf9, 64, 4, grid=grid(64, 4), stream=stream0)
        buf10 = reinterpret_tensor(buf4, (64, 4), (4, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf8, (64, 4), (4, 1), 0), reinterpret_tensor(primals_9, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf10)
        del primals_10
    return (reinterpret_tensor(buf10, (4, 4, 4, 4), (64, 16, 4, 1), 0), reinterpret_tensor(primals_3, (64, 4), (4, 1), 0), reinterpret_tensor(primals_6, (64, 4), (4, 1), 0), buf7, reinterpret_tensor(buf2, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf3, (16, 4, 4), (16, 1, 4), 0), buf9, reinterpret_tensor(buf8, (64, 4), (4, 1), 0), primals_9, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
