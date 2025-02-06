# AOT ID: ['3_inference']
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


# kernel path: inductor_cache/lf/clf7jgxgqpga3lbslctp5blxb5exy6nybwqetz6gdeme4isrh3oc.py
# Topologically Sorted Source Nodes: [bases], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   bases => repeat
# Graph fragment:
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%arg1_1, [4, 1, 1]), kwargs = {})
triton_poi_fused_repeat_0 = async_compile.triton('triton_poi_fused_repeat_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/a2/ca2d2xszggujgcynqns6zm4uzkoj76jdll7dlfjnta2jyws2uktt.py
# Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attention_1 => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_4, [2], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_4, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
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
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cw/ccwvohvgewewtgxgoi2subksbtgow2heh7aqviuui7vy4smp5zu2.py
# Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attention_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [2], True), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_2 = async_compile.triton('triton_poi_fused__softmax_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/cg/ccg7o3ubteaiwnnmnsa5tdn3sye5w4dblyb2mwt7freszkggxof5.py
# Topologically Sorted Source Nodes: [attention_normed], Original ATen: [aten.linalg_vector_norm]
# Source node to ATen node mapping:
#   attention_normed => abs_1, sum_2
# Graph fragment:
#   %abs_1 : [num_users=1] = call_function[target=torch.ops.aten.abs.default](args = (%div,), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%abs_1, [1], True), kwargs = {})
triton_per_fused_linalg_vector_norm_3 = async_compile.triton('triton_per_fused_linalg_vector_norm_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 16, 'r': 16},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_linalg_vector_norm_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_linalg_vector_norm_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*r2 + 64*x1), xmask, other=0.0)
    tmp1 = tl_math.abs(tmp0)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pe/cpeodhk3t4khjbk5xsheltexlev2in5r472fhombowq4zzuol34g.py
# Topologically Sorted Source Nodes: [attention_normed], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   attention_normed => div_1
# Graph fragment:
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div, %expand), kwargs = {})
triton_poi_fused_div_4 = async_compile.triton('triton_poi_fused_div_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = 1e-12
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp0 / tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/l6/cl6xsxbv5kjxilzo32jcg53yzwfqy2sjhelds456jcveuarwsa3v.py
# Topologically Sorted Source Nodes: [bases_2], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   bases_2 => div_2
# Graph fragment:
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%view_8, %expand_1), kwargs = {})
triton_poi_fused_div_5 = async_compile.triton('triton_poi_fused_div_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (4 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (8 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (12 + x0 + 16*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp1 * tmp1
    tmp4 = tmp3 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6 * tmp6
    tmp8 = tmp5 + tmp7
    tmp10 = tmp9 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = 1e-12
    tmp14 = triton_helpers.maximum(tmp12, tmp13)
    tmp15 = tmp0 / tmp14
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e3/ce3mtqw6zvdgs2j5v3xxuobgks455kkhxd5gh7lgmyvnk3yeo255.py
# Topologically Sorted Source Nodes: [attention_normed_3], Original ATen: [aten.div]
# Source node to ATen node mapping:
#   attention_normed_3 => div_10
# Graph fragment:
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%div_9, %expand_6), kwargs = {})
triton_poi_fused_div_6 = async_compile.triton('triton_poi_fused_div_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x2 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 4*x2), xmask, eviction_policy='evict_last')
    tmp2 = 1e-12
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (1, 4, 4), (16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bases], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_0.run(arg1_1, buf0, 64, grid=grid(64), stream=stream0)
        del arg1_1
        buf1 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 16, 4), (64, 1, 16), 0), buf0, out=buf1)
        buf2 = empty_strided_cuda((4, 16, 4), (64, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf1, buf2, 256, grid=grid(256), stream=stream0)
        buf3 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [attention_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf2, buf3, 256, grid=grid(256), stream=stream0)
        buf4 = empty_strided_cuda((4, 1, 4), (4, 16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_normed], Original ATen: [aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_3.run(buf3, buf4, 16, 16, grid=grid(16), stream=stream0)
        buf5 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [attention_normed], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_4.run(buf5, buf4, 256, grid=grid(256), stream=stream0)
        buf6 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [bases_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 4, 16), (64, 16, 1), 0), buf5, out=buf6)
        buf7 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [bases_2], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf6, buf7, 64, grid=grid(64), stream=stream0)
        buf8 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [attention_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 16, 4), (64, 1, 16), 0), buf7, out=buf8)
        buf9 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [attention_3], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf8, buf9, 256, grid=grid(256), stream=stream0)
        buf10 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [attention_3], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf9, buf10, 256, grid=grid(256), stream=stream0)
        buf11 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [attention_normed_1], Original ATen: [aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_3.run(buf10, buf11, 16, 16, grid=grid(16), stream=stream0)
        buf12 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [attention_normed_1], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_4.run(buf12, buf11, 256, grid=grid(256), stream=stream0)
        buf13 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [bases_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 4, 16), (64, 16, 1), 0), buf12, out=buf13)
        buf14 = buf6; del buf6  # reuse
        # Topologically Sorted Source Nodes: [bases_4], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf13, buf14, 64, grid=grid(64), stream=stream0)
        buf15 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [attention_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 16, 4), (64, 1, 16), 0), buf14, out=buf15)
        buf16 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [attention_5], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf15, buf16, 256, grid=grid(256), stream=stream0)
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [attention_5], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf16, buf17, 256, grid=grid(256), stream=stream0)
        buf18 = buf11; del buf11  # reuse
        # Topologically Sorted Source Nodes: [attention_normed_2], Original ATen: [aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_3.run(buf17, buf18, 16, 16, grid=grid(16), stream=stream0)
        buf19 = buf17; del buf17  # reuse
        # Topologically Sorted Source Nodes: [attention_normed_2], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_4.run(buf19, buf18, 256, grid=grid(256), stream=stream0)
        buf20 = buf14; del buf14  # reuse
        # Topologically Sorted Source Nodes: [bases_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 4, 16), (64, 16, 1), 0), buf19, out=buf20)
        buf21 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [bases_6], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf20, buf21, 64, grid=grid(64), stream=stream0)
        buf22 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [attention_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 16, 4), (64, 1, 16), 0), buf21, out=buf22)
        buf23 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [attention_7], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_1.run(buf22, buf23, 256, grid=grid(256), stream=stream0)
        buf24 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [attention_7], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_2.run(buf23, buf24, 256, grid=grid(256), stream=stream0)
        buf25 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [attention_normed_3], Original ATen: [aten.linalg_vector_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused_linalg_vector_norm_3.run(buf24, buf25, 16, 16, grid=grid(16), stream=stream0)
        buf26 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [attention_normed_3], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_6.run(buf24, buf25, buf26, 256, grid=grid(256), stream=stream0)
        del buf25
        buf27 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [bases_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(arg0_1, (4, 4, 16), (64, 16, 1), 0), buf26, out=buf27)
        del arg0_1
        buf28 = buf20; del buf20  # reuse
        # Topologically Sorted Source Nodes: [bases_8], Original ATen: [aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_5.run(buf27, buf28, 64, grid=grid(64), stream=stream0)
        del buf27
        buf29 = reinterpret_tensor(buf26, (4, 4, 16), (64, 16, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [feats_recon], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf28, reinterpret_tensor(buf24, (4, 4, 16), (64, 1, 4), 0), out=buf29)
        del buf24
        del buf28
    return (reinterpret_tensor(buf29, (4, 4, 4, 4), (64, 16, 4, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
