# AOT ID: ['12_forward']
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


# kernel path: inductor_cache/5o/c5oiqg5ngoup2p2o2qgmtn7kpg7pmmto67ry3qwcxblezxec7gow.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %slice_2), kwargs = {})
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tu/ctu6wffnrqja5gocafc6ggbnxn76al2nrojoybom3njigkzr37vg.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   multi_head_attention_forward => mul
# Graph fragment:
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%view_9, 1.0), kwargs = {})
#   %permute_26 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_12, [0, 2, 1]), kwargs = {})
triton_poi_fused_mul_transpose_1 = async_compile.triton('triton_poi_fused_mul_transpose_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_transpose_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_transpose_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (y3 + 16*x2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
    tl.store(out_ptr1 + (y3 + 16*x2), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/fx/cfx5ckgpaympkesfveqa5c7fev7b3p6soasyrc5e3pos6smgbyp2.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul, aten.transpose]
# Source node to ATen node mapping:
#   multi_head_attention_forward => mul_1
# Graph fragment:
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_6, 1.0), kwargs = {})
#   %permute_27 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%view_13, [0, 2, 1]), kwargs = {})
triton_poi_fused_mul_transpose_2 = async_compile.triton('triton_poi_fused_mul_transpose_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_transpose_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_transpose_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (y3 + 16*x2), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4 + y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
    tl.store(out_ptr1 + (y3 + 16*x2), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ma/cmacacwnrfcpr6rzqkfrkjunlw6sqthsjr4pg7zbdjd4r2vzw6ci.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   multi_head_attention_forward => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_14, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_14, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/h2/ch2n5u2ub2c6yxsk7rmxpwwtcv4uk2t2stewv237q35zmlsuqi7c.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._safe_softmax]
# Source node to ATen node mapping:
#   multi_head_attention_forward => any_1, div, eq, full_default, logical_not, logical_not_1, sum_1, where
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%view_14, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 4, 4, 4], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default, %div), kwargs = {})
triton_poi_fused__safe_softmax_4 = async_compile.triton('triton_poi_fused__safe_softmax_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__safe_softmax_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__safe_softmax_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/ex/cexieuifxu3pemcju2csk6cinru6vrh6y7o7clul3xk5cqpybokx.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_7,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_5 = async_compile.triton('triton_poi_fused_clone_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 4, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 16*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ih/cih6z5taqjm5rvtxuqcbuw2qs2qju6rc5n4qkil6eggmnnfd2pmx.py
# Topologically Sorted Source Nodes: [add_2, x_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_2 => add_2
#   x_2 => var_mean
# Graph fragment:
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_19), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_2, [2]), kwargs = {correction: 0, keepdim: True})
triton_poi_fused_add_native_layer_norm_6 = async_compile.triton('triton_poi_fused_add_native_layer_norm_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_6(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (4*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (4*x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (1 + 4*x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (2 + 4*x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (3 + 4*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 + tmp13
    tmp15 = 4.0
    tmp16 = tmp14 / tmp15
    tmp17 = tmp2 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tmp5 - tmp16
    tmp20 = tmp19 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp9 - tmp16
    tmp23 = tmp22 * tmp22
    tmp24 = tmp21 + tmp23
    tmp25 = tmp13 - tmp16
    tmp26 = tmp25 * tmp25
    tmp27 = tmp24 + tmp26
    tmp28 = tmp27 / tmp15
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp28, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/on/conq2m7yrzjahilft3znmykt5533eci5z6jh32ubyu65vpe3fezz.py
# Topologically Sorted Source Nodes: [add_2, x_2], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_2 => add_2
#   x_2 => add_3, add_4, mul_2, mul_3, rsqrt, sub_1
# Graph fragment:
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%primals_1, %view_19), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_2, %getitem_7), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %primals_7), kwargs = {})
#   %add_4 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %primals_8), kwargs = {})
triton_poi_fused_add_native_layer_norm_7 = async_compile.triton('triton_poi_fused_add_native_layer_norm_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 4
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
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


# kernel path: inductor_cache/4l/c4lvnsdhk5xbzqg6agtcnxbs7z5h5qqkat3y4lhkmpi6o4vw23kf.py
# Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
# Source node to ATen node mapping:
#   gelu => add_5, erf, mul_4, mul_5, mul_6
# Graph fragment:
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.5), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_21, 0.7071067811865476), kwargs = {})
#   %erf : [num_users=1] = call_function[target=torch.ops.aten.erf.default](args = (%mul_5,), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%erf, 1), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %add_5), kwargs = {})
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
    xnumel = 32768
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


# kernel path: inductor_cache/zt/cztazqwaffwmuyripttmidcvo5oa74s2r56eotmls4bjsslgzg6c.py
# Topologically Sorted Source Nodes: [add_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_3 => add_6
# Graph fragment:
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %view_23), kwargs = {})
triton_poi_fused_add_9 = async_compile.triton('triton_poi_fused_add_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_9(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k7/ck7f2yo2wqf4nee4xnwm3q4lrv5nvlqoh5tid74gxl67b3zw4dy3.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_4 => add_7, rsqrt_1, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
triton_poi_fused_native_layer_norm_10 = async_compile.triton('triton_poi_fused_native_layer_norm_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_10(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/sf/csflrlgnice6z5oxu4wz5jolkguykgsp5wnhrglkafvgelljhx7s.py
# Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   x_4 => add_7, add_8, mul_7, mul_8, rsqrt_1, sub_2, var_mean_1
# Graph fragment:
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_8, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_9), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %rsqrt_1), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %primals_13), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %primals_14), kwargs = {})
triton_poi_fused_native_layer_norm_11 = async_compile.triton('triton_poi_fused_native_layer_norm_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (1, 2048, 4), (8192, 4, 1))
    assert_size_stride(primals_3, (12, 4), (4, 1))
    assert_size_stride(primals_4, (12, ), (1, ))
    assert_size_stride(primals_5, (4, 4), (4, 1))
    assert_size_stride(primals_6, (4, ), (1, ))
    assert_size_stride(primals_7, (4, ), (1, ))
    assert_size_stride(primals_8, (4, ), (1, ))
    assert_size_stride(primals_9, (2048, 4), (4, 1))
    assert_size_stride(primals_10, (2048, ), (1, ))
    assert_size_stride(primals_11, (4, 2048), (2048, 1))
    assert_size_stride(primals_12, (4, ), (1, ))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(primals_1, primals_2, buf0, 64, grid=grid(64), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (16, 4), (4, 1), 0), reinterpret_tensor(primals_3, (4, 4), (1, 4), 0), out=buf1)
        buf2 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (16, 4), (4, 1), 0), reinterpret_tensor(primals_3, (4, 4), (1, 4), 16), out=buf2)
        buf3 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.addmm(reinterpret_tensor(primals_4, (4, ), (1, ), 8), reinterpret_tensor(primals_1, (16, 4), (4, 1), 0), reinterpret_tensor(primals_3, (4, 4), (1, 4), 32), alpha=1, beta=1, out=buf3)
        del primals_3
        buf4 = empty_strided_cuda((4, 4, 4, 1), (16, 4, 1, 1), torch.float32)
        buf22 = empty_strided_cuda((16, 1, 4), (1, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_transpose_1.run(buf1, primals_4, buf4, buf22, 16, 4, grid=grid(16, 4), stream=stream0)
        buf5 = reinterpret_tensor(buf1, (4, 4, 1, 4), (16, 4, 4, 1), 0); del buf1  # reuse
        buf23 = empty_strided_cuda((16, 4, 1), (1, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mul, aten.transpose]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_transpose_2.run(buf2, primals_4, buf5, buf23, 16, 4, grid=grid(16, 4), stream=stream0)
        del primals_4
        buf6 = empty_strided_cuda((16, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf4, (16, 4, 1), (4, 1, 0), 0), reinterpret_tensor(buf5, (16, 1, 4), (4, 0, 1), 0), out=buf6)
        buf7 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_3.run(buf6, buf7, 256, grid=grid(256), stream=stream0)
        buf8 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__safe_softmax_4.run(buf6, buf7, buf8, 256, grid=grid(256), stream=stream0)
        del buf6
        del buf7
        buf9 = reinterpret_tensor(buf5, (16, 4, 1), (4, 1, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf8, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf3, (16, 4, 1), (1, 16, 0), 0), out=buf9)
        buf10 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_5.run(buf9, buf10, 4, 16, grid=grid(4, 16), stream=stream0)
        buf11 = reinterpret_tensor(buf9, (16, 4), (4, 1), 0); del buf9  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_6, reinterpret_tensor(buf10, (16, 4), (4, 1), 0), reinterpret_tensor(primals_5, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf11)
        del primals_6
        buf12 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf13 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [add_2, x_2], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_6.run(primals_1, buf11, buf12, buf13, 16, grid=grid(16), stream=stream0)
        buf14 = reinterpret_tensor(buf2, (4, 4, 4), (16, 4, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [add_2, x_2], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_7.run(primals_1, buf11, buf12, buf13, primals_7, primals_8, buf14, 64, grid=grid(64), stream=stream0)
        del primals_8
        buf15 = empty_strided_cuda((16, 2048), (2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_10, reinterpret_tensor(buf14, (16, 4), (4, 1), 0), reinterpret_tensor(primals_9, (4, 2048), (1, 4), 0), alpha=1, beta=1, out=buf15)
        del primals_10
        buf16 = empty_strided_cuda((4, 4, 2048), (8192, 2048, 1), torch.float32)
        # Topologically Sorted Source Nodes: [gelu], Original ATen: [aten.gelu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_gelu_8.run(buf15, buf16, 32768, grid=grid(32768), stream=stream0)
        buf17 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf16, (16, 2048), (2048, 1), 0), reinterpret_tensor(primals_11, (2048, 4), (1, 2048), 0), out=buf17)
        buf18 = reinterpret_tensor(buf17, (4, 4, 4), (16, 4, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [add_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_9.run(buf18, buf14, primals_12, 64, grid=grid(64), stream=stream0)
        del primals_12
        buf19 = buf13; del buf13  # reuse
        buf20 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_10.run(buf18, buf19, buf20, 16, grid=grid(16), stream=stream0)
        buf21 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_11.run(buf18, buf19, buf20, primals_13, primals_14, buf21, 64, grid=grid(64), stream=stream0)
        del buf19
        del buf20
        del primals_14
    return (buf21, primals_1, primals_7, primals_13, reinterpret_tensor(buf0, (16, 4), (4, 1), 0), buf8, reinterpret_tensor(buf10, (16, 4), (4, 1), 0), buf11, reinterpret_tensor(buf14, (16, 4), (4, 1), 0), buf15, reinterpret_tensor(buf16, (16, 2048), (2048, 1), 0), buf18, primals_11, primals_9, primals_5, reinterpret_tensor(buf3, (16, 1, 4), (1, 1, 16), 0), buf22, buf23, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, 2048, 4), (8192, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((12, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((2048, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
