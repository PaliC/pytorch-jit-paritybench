# AOT ID: ['14_forward']
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


# kernel path: inductor_cache/zq/czqpy4zp2ef22jm66kvpkocxc4qkt334oftrozb7z6fpcqs346zk.py
# Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh]
# Source node to ATen node mapping:
#   setitem_1 => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([], 1.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_lift_fresh_0 = async_compile.triton('triton_poi_fused_lift_fresh_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0,), 'tt.equal_to': (1,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_lift_fresh_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_lift_fresh_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = 1.0
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/e4/ce4mspsihehuwzw5ebzo2b3jxntrlfi3n7m5w6spdswlwu2d3cfp.py
# Topologically Sorted Source Nodes: [softplus, add], Original ATen: [aten.softplus, aten.add]
# Source node to ATen node mapping:
#   add => add
#   softplus => exp, gt, log1p, where
# Graph fragment:
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%primals_8,), kwargs = {})
#   %log1p : [num_users=1] = call_function[target=torch.ops.aten.log1p.default](args = (%exp,), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%primals_8, 20), kwargs = {})
#   %where : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%gt, %primals_8, %log1p), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%where, 0.001), kwargs = {})
triton_poi_fused_add_softplus_1 = async_compile.triton('triton_poi_fused_add_softplus_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_softplus_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_softplus_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = 20.0
    tmp2 = tmp0 > tmp1
    tmp3 = tl_math.exp(tmp0)
    tmp4 = libdevice.log1p(tmp3)
    tmp5 = tl.where(tmp2, tmp0, tmp4)
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tl.store(out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6r/c6r5c4wfdmfwadc67irjvsb6e6kez2iog2oojzoghvfedeeqcgxi.py
# Topologically Sorted Source Nodes: [log, sum_1, logabsdet, new_ones, logabsdet_1, view], Original ATen: [aten.log, aten.sum, aten.neg, aten.new_ones, aten.mul, aten.view]
# Source node to ATen node mapping:
#   log => log
#   logabsdet => neg
#   logabsdet_1 => mul
#   new_ones => full_default_3
#   sum_1 => sum_1
#   view => view
# Graph fragment:
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%add,), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.default](args = (%log,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sum_1,), kwargs = {})
#   %full_default_3 : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4], 1), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%neg, %full_default_3), kwargs = {})
#   %view : [num_users=1] = call_function[target=torch.ops.aten.reshape.default](args = (%mul, [-1]), kwargs = {})
triton_poi_fused_log_mul_neg_new_ones_sum_view_2 = async_compile.triton('triton_poi_fused_log_mul_neg_new_ones_sum_view_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_log_mul_neg_new_ones_sum_view_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_log_mul_neg_new_ones_sum_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr0 + (1))
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK])
    tmp7 = tl.load(in_ptr0 + (2))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK])
    tmp11 = tl.load(in_ptr0 + (3))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK])
    tmp2 = tl_math.log(tmp1)
    tmp5 = tl_math.log(tmp4)
    tmp6 = tmp2 + tmp5
    tmp9 = tl_math.log(tmp8)
    tmp10 = tmp6 + tmp9
    tmp13 = tl_math.log(tmp12)
    tmp14 = tmp10 + tmp13
    tmp15 = -tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr0 + (x0), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ci/cciwqi4qbvphjyx33wydhunyo4nl7d73z6blzeta3mrxg7ifxet6.py
# Topologically Sorted Source Nodes: [argsort], Original ATen: [aten.sort]
# Source node to ATen node mapping:
#   argsort => getitem_1, sort
# Graph fragment:
#   %sort : [num_users=1] = call_function[target=torch.ops.aten.sort.default](args = (%primals_11,), kwargs = {})
#   %getitem_1 : [num_users=2] = call_function[target=operator.getitem](args = (%sort, 1), kwargs = {})
triton_per_fused_sort_3 = async_compile.triton('triton_per_fused_sort_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1, 'r': 4},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*i64', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': (2,)}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sort_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_sort_3(in_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), None)
    tmp1 = r0
    tmp2 = tmp1.to(tl.int16)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5, tmp6, = triton_helpers.sort_with_index(tmp3, tmp4, None, 1, stable=False, descending=False)
    tmp7 = tmp6.to(tl.int64)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp7, None)
''', device_str='cuda')


# kernel path: inductor_cache/dh/cdh5ciilbaf5lc7525eeixdxyt6fh3ww6pau667cjr7rpuwifocr.py
# Topologically Sorted Source Nodes: [lower], Original ATen: [aten.new_zeros]
# Source node to ATen node mapping:
#   lower => full_default
# Graph fragment:
#   %full_default : [num_users=3] = call_function[target=torch.ops.aten.full.default](args = ([4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_new_zeros_4 = async_compile.triton('triton_poi_fused_new_zeros_4', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_new_zeros_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_new_zeros_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vu/cvulftdagkutgzqvslsqgybmfismterfvy5pdf4rqoqqzxgavj2a.py
# Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   outputs => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_10, %primals_9), kwargs = {})
triton_poi_fused_sub_5 = async_compile.triton('triton_poi_fused_sub_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sub_5(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/di/cdikxltasxq3wqtou2i7oh4xfuaq5d2bknoucrhndoqggxuhojjj.py
# Topologically Sorted Source Nodes: [outputs_4], Original ATen: [aten.index_select]
# Source node to ATen node mapping:
#   outputs_4 => index
# Graph fragment:
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%permute_1, [None, %getitem_1]), kwargs = {})
triton_poi_fused_index_select_6 = async_compile.triton('triton_poi_fused_index_select_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_select_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_index_select_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = xindex // 4
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 4, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert(((0 <= tmp4) & (tmp4 < 4)) | ~(xmask), "index out of bounds: 0 <= tmp4 < 4")
    tmp6 = tl.load(in_ptr1 + (tmp4 + 4*x1), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11 = args
    args.clear()
    assert_size_stride(primals_1, (6, ), (1, ))
    assert_size_stride(primals_2, (6, ), (1, ))
    assert_size_stride(primals_3, (6, ), (1, ))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (6, ), (1, ))
    assert_size_stride(primals_6, (6, ), (1, ))
    assert_size_stride(primals_7, (6, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (4, 4), (4, 1))
    assert_size_stride(primals_11, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((), (), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.lift_fresh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_lift_fresh_0.run(buf3, 1, grid=grid(1), stream=stream0)
        buf7 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [softplus, add], Original ATen: [aten.softplus, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_softplus_1.run(primals_8, buf7, 4, grid=grid(4), stream=stream0)
        buf18 = empty_strided_cuda((4, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [log, sum_1, logabsdet, new_ones, logabsdet_1, view], Original ATen: [aten.log, aten.sum, aten.neg, aten.new_ones, aten.mul, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_log_mul_neg_new_ones_sum_view_2.run(buf7, buf18, 4, grid=grid(4), stream=stream0)
        buf16 = empty_strided_cuda((4, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [argsort], Original ATen: [aten.sort]
        stream0 = get_raw_stream(0)
        triton_per_fused_sort_3.run(primals_11, buf16, 1, 4, grid=grid(1), stream=stream0)
        del primals_11
        buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [lower], Original ATen: [aten.new_zeros]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_4.run(buf0, 16, grid=grid(16), stream=stream0)
        buf1 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_4.run(buf1, 16, grid=grid(16), stream=stream0)
        aten.index_put_(buf1, [primals_2, primals_3], primals_1, False)
        del primals_1
        aten.index_put_(buf1, [primals_4, primals_4], buf3, False)
        del buf3
        buf5 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.index_put]
        stream0 = get_raw_stream(0)
        triton_poi_fused_new_zeros_4.run(buf5, 16, grid=grid(16), stream=stream0)
        aten.index_put_(buf5, [primals_6, primals_7], primals_5, False)
        del primals_5
        aten.index_put_(buf5, [primals_4, primals_4], buf7, False)
        del buf7
        buf9 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [outputs], Original ATen: [aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sub_5.run(primals_10, primals_9, buf9, 16, grid=grid(16), stream=stream0)
        del primals_10
        del primals_9
        # Topologically Sorted Source Nodes: [outputs_1], Original ATen: [aten.linalg_solve_triangular]
        buf10 = torch.ops.aten.linalg_solve_triangular.default(buf1, reinterpret_tensor(buf9, (4, 4), (1, 4), 0), upper=False, unitriangular=True)
        buf11 = buf10
        del buf10
        # Topologically Sorted Source Nodes: [outputs_2], Original ATen: [aten.linalg_solve_triangular]
        buf12 = torch.ops.aten.linalg_solve_triangular.default(buf5, buf11, upper=True)
        buf13 = buf12
        del buf12
        buf17 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [outputs_4], Original ATen: [aten.index_select]
        stream0 = get_raw_stream(0)
        triton_poi_fused_index_select_6.run(buf16, buf13, buf17, 16, grid=grid(16), stream=stream0)
    return (buf17, buf18, primals_2, primals_3, primals_4, primals_6, primals_7, primals_8, buf0, reinterpret_tensor(buf13, (4, 4), (4, 1), 0), buf16, reinterpret_tensor(buf5, (4, 4), (1, 4), 0), reinterpret_tensor(buf1, (4, 4), (1, 4), 0), reinterpret_tensor(buf11, (4, 4), (4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.int64)
    primals_3 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.int64)
    primals_4 = rand_strided((4, ), (1, ), device='cpu', dtype=torch.int64)
    primals_5 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.int64)
    primals_7 = rand_strided((6, ), (1, ), device='cpu', dtype=torch.int64)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.int64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
