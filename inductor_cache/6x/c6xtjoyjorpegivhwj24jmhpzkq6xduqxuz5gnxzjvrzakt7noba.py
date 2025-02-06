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


# kernel path: inductor_cache/yo/cyofsbkgqaevvhdvmxyhmaxgaiicq2myopeddsnpwfiywj2wdub5.py
# Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   matmul => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_0 = async_compile.triton('triton_poi_fused_clone_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 4*y3), tmp2, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jx/cjxdcsft6rhwa74jye5n5auvkmi74tng6v7j5s3wj3v4wog2aw7l.py
# Topologically Sorted Source Nodes: [eq], Original ATen: [aten.eq]
# Source node to ATen node mapping:
#   eq => eq
# Graph fragment:
#   %eq : [num_users=2] = call_function[target=torch.ops.aten.eq.Scalar](args = (%unsqueeze, 0), kwargs = {})
triton_poi_fused_eq_1 = async_compile.triton('triton_poi_fused_eq_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_eq_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_eq_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 0.0
    tmp2 = tmp0 == tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xw/cxwi4jgiblzhi2gzgdqrp3oz43pas3rju2idgyz35j4dj57txsml.py
# Topologically Sorted Source Nodes: [scores, scores_1, p_attn], Original ATen: [aten.div, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   p_attn => amax, exp, sub, sum_1
#   scores => div
#   scores_1 => full_default, where
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_11, 1.0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1000000000.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %div), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
triton_poi_fused__softmax_div_masked_fill_2 = async_compile.triton('triton_poi_fused__softmax_div_masked_fill_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_div_masked_fill_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_div_masked_fill_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x2 = xindex // 16
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0 + 16*x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (4*x3), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (1 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp7 = tl.load(in_ptr1 + (1 + 4*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (2 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp12 = tl.load(in_ptr1 + (2 + 4*x3), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr0 + (3 + 4*x0 + 16*x2), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp17 = tl.load(in_ptr1 + (3 + 4*x3), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = -1000000000.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp8 = tmp7 * tmp2
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = triton_helpers.maximum(tmp5, tmp9)
    tmp13 = tmp12 * tmp2
    tmp14 = tl.where(tmp11, tmp4, tmp13)
    tmp15 = triton_helpers.maximum(tmp10, tmp14)
    tmp18 = tmp17 * tmp2
    tmp19 = tl.where(tmp16, tmp4, tmp18)
    tmp20 = triton_helpers.maximum(tmp15, tmp19)
    tmp21 = tmp5 - tmp20
    tmp22 = tl_math.exp(tmp21)
    tmp23 = tmp9 - tmp20
    tmp24 = tl_math.exp(tmp23)
    tmp25 = tmp22 + tmp24
    tmp26 = tmp14 - tmp20
    tmp27 = tl_math.exp(tmp26)
    tmp28 = tmp25 + tmp27
    tmp29 = tmp19 - tmp20
    tmp30 = tl_math.exp(tmp29)
    tmp31 = tmp28 + tmp30
    tl.store(out_ptr0 + (x3), tmp20, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fu/cfu2ccig4p7dnwzeqmh5szwogavjbmb4fomikb7fgfof7d6u2vpm.py
# Topologically Sorted Source Nodes: [scores, scores_1, p_attn], Original ATen: [aten.div, aten.masked_fill, aten._softmax]
# Source node to ATen node mapping:
#   p_attn => amax, div_1, exp, sub
#   scores => div
#   scores_1 => full_default, where
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_11, 1.0), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], -1000000000.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%eq, %full_default, %div), kwargs = {})
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%where, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%where, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_div_masked_fill_3 = async_compile.triton('triton_poi_fused__softmax_div_masked_fill_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i1', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_div_masked_fill_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_div_masked_fill_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex // 64
    x4 = (xindex % 16)
    x5 = xindex
    x6 = xindex // 4
    tmp0 = tl.load(in_ptr0 + (x4 + 16*x3), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tl.load(in_out_ptr0 + (x5), xmask)
    tmp6 = tl.load(in_ptr1 + (x6), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x6), xmask, eviction_policy='evict_last')
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = -1000000000.0
    tmp5 = tl.where(tmp0, tmp4, tmp3)
    tmp7 = tmp5 - tmp6
    tmp8 = tl_math.exp(tmp7)
    tmp10 = tmp8 / tmp9
    tl.store(in_out_ptr0 + (x5), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/an/cany2jbw2jdajpiihiclpvumcasljofi46nmewpzgmcfjpkuy5qh.py
# Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   contiguous => clone_4
# Graph fragment:
#   %clone_4 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    assert_size_stride(primals_4, (4, 4), (4, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_7, (4, 4), (4, 1))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_10, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_11, (4, 4), (4, 1))
    assert_size_stride(primals_12, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_1, (16, 4), (4, 1), 0), reinterpret_tensor(primals_2, (4, 4), (1, 4), 0), out=buf0)
        del primals_2
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_6, (16, 4), (4, 1), 0), reinterpret_tensor(primals_4, (4, 4), (1, 4), 0), out=buf1)
        del primals_4
        buf2 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(primals_9, (16, 4), (4, 1), 0), reinterpret_tensor(primals_7, (4, 4), (1, 4), 0), out=buf2)
        del primals_7
        buf3 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf0, primals_3, buf3, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_3
        buf4 = reinterpret_tensor(buf0, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf1, primals_5, buf4, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_5
        buf5 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf3, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf4, (16, 1, 4), (4, 0, 1), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 1, 4, 4), (16, 16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [eq], Original ATen: [aten.eq]
        stream0 = get_raw_stream(0)
        triton_poi_fused_eq_1.run(primals_10, buf6, 64, grid=grid(64), stream=stream0)
        del primals_10
        buf7 = reinterpret_tensor(buf1, (4, 4, 4, 1), (16, 4, 1, 64), 0); del buf1  # reuse
        buf8 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 64), torch.float32)
        # Topologically Sorted Source Nodes: [scores, scores_1, p_attn], Original ATen: [aten.div, aten.masked_fill, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_masked_fill_2.run(buf6, buf5, buf7, buf8, 64, grid=grid(64), stream=stream0)
        buf9 = reinterpret_tensor(buf5, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [scores, scores_1, p_attn], Original ATen: [aten.div, aten.masked_fill, aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_div_masked_fill_3.run(buf9, buf6, buf7, buf8, 256, grid=grid(256), stream=stream0)
        buf10 = reinterpret_tensor(buf8, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_0.run(buf2, primals_8, buf10, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_8
        buf11 = reinterpret_tensor(buf2, (16, 4, 1), (4, 1, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf9, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf10, (16, 4, 1), (4, 1, 0), 0), out=buf11)
        buf12 = reinterpret_tensor(buf7, (4, 4, 4, 1), (16, 4, 1, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_4.run(buf11, buf12, 16, 4, grid=grid(16, 4), stream=stream0)
        buf13 = reinterpret_tensor(buf11, (16, 4), (4, 1), 0); del buf11  # reuse
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_12, reinterpret_tensor(buf12, (16, 4), (4, 1), 0), reinterpret_tensor(primals_11, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf13)
        del primals_12
    return (reinterpret_tensor(buf13, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(primals_1, (16, 4), (4, 1), 0), reinterpret_tensor(primals_6, (16, 4), (4, 1), 0), reinterpret_tensor(primals_9, (16, 4), (4, 1), 0), buf6, buf9, reinterpret_tensor(buf12, (16, 4), (4, 1), 0), primals_11, reinterpret_tensor(buf10, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf3, (16, 1, 4), (4, 1, 1), 0), reinterpret_tensor(buf4, (16, 4, 1), (4, 1, 4), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
