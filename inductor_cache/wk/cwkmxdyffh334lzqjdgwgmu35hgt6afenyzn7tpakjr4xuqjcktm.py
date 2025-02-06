# AOT ID: ['1_forward']
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


# kernel path: inductor_cache/hb/chbmzlsmkqw5h6jhwu3pixbo72roe4adygaxkscj4j7kkkoktzn6.py
# Topologically Sorted Source Nodes: [X], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   X => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_2, %primals_1], 2), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (4*x1 + ((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ky/cky57ryzjzqyo32o7jyzjhxoajcwpgcnbeenhiqp7bkjnbvuf2mp.py
# Topologically Sorted Source Nodes: [x_with_pos], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_with_pos => add
# Graph fragment:
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %primals_5), kwargs = {})
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 4)
    x4 = (xindex % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/3h/c3hy6sgxsldjmhpdvam3uf5gbtcrtal74ctml5w5jonbeflyt6ob.py
# Topologically Sorted Source Nodes: [truediv, attn], Original ATen: [aten.div, aten.clone]
# Source node to ATen node mapping:
#   attn => clone_2
#   truediv => div
# Graph fragment:
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%permute_4, 2.0), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_div_2 = async_compile.triton('triton_poi_fused_clone_div_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_div_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_div_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 16*x1 + 64*x3), xmask)
    tmp1 = 0.5
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wi/cwixm4jzpwbjhmpe6vctjpuvbjx7iitzmcxukgbidnwf2ph4s6vr.py
# Topologically Sorted Source Nodes: [attn], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attn => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 16*x2 + 64*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/op/copox25w4pf2us4f7tzlpcb4d467cnqap2ty3f5nxw67sz54zviq.py
# Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attn_1 => amax, exp, sub
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_13, [-1], True), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_13, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub,), kwargs = {})
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
#   attn_1 => div_1, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [-1], True), kwargs = {})
#   %div_1 : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
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


# kernel path: inductor_cache/4t/c4t4kevu4476fuvcmfsmvsfqlxvjb77otza7y2tl4gbmdsvw4uls.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   output => clone_5
# Graph fragment:
#   %clone_5 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 4)
    x1 = ((xindex // 4) % 4)
    x2 = ((xindex // 16) % 4)
    x3 = xindex // 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 4*x2 + 16*x1 + 64*x3), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jj/cjj7m2iqiwepnabb5frmsfb3g6hfzerwai4mb5si6ngyu2lnhck3.py
# Topologically Sorted Source Nodes: [enc_output_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   enc_output_1 => add_1
# Graph fragment:
#   %add_1 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_19, %add), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/53/c53pedtacqbifhouwju3zdtcimdhidnlvdyyf34thyptttvmc5ek.py
# Topologically Sorted Source Nodes: [enc_output_2], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   enc_output_2 => add_2, rsqrt, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
triton_poi_fused_native_layer_norm_8 = async_compile.triton('triton_poi_fused_native_layer_norm_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_8(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp21 = 1e-06
    tmp22 = tmp20 + tmp21
    tmp23 = libdevice.rsqrt(tmp22)
    tl.store(out_ptr0 + (x0), tmp8, xmask)
    tl.store(out_ptr1 + (x0), tmp23, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/cc/cccxcqbj4xb2fe6423h2ejlvh2e7y7l2uitxwkl762ew7g7ygbfb.py
# Topologically Sorted Source Nodes: [enc_output_2], Original ATen: [aten.native_layer_norm]
# Source node to ATen node mapping:
#   enc_output_2 => add_2, add_3, mul, mul_1, rsqrt, sub_1, var_mean
# Graph fragment:
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_10), kwargs = {})
#   %add_3 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_11), kwargs = {})
triton_poi_fused_native_layer_norm_9 = async_compile.triton('triton_poi_fused_native_layer_norm_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/wq/cwqnkef2ejwkakzoynqjtvvermxcrjibba2u6nkzdwzkpxidhimz.py
# Topologically Sorted Source Nodes: [x], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   x => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_21,), kwargs = {})
#   %le_2 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_10 = async_compile.triton('triton_poi_fused_relu_threshold_backward_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_10(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ob/cobcfn3pe5youysdcxeyiopmdbf2w3iy5wi2rytz73s7msyey5la.py
# Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   x_3 => add_4
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_23, %add_3), kwargs = {})
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_11', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_11(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ks/cks4pgexpxmb6xobhvitjpsfme7ebgff5da52ncb4ezlxvt6dgw3.py
# Topologically Sorted Source Nodes: [X_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   X_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7, %primals_1], 2), kwargs = {})
triton_poi_fused_cat_12 = async_compile.triton('triton_poi_fused_cat_12', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x1 = xindex // 8
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 * tmp6
    tmp8 = 1.0
    tmp9 = tmp8 - tmp5
    tmp10 = tl.load(in_ptr2 + (4*x1 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp7 + tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp4, tmp12, tmp13)
    tmp15 = tmp0 >= tmp3
    tmp16 = tl.full([1], 8, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr0 + (4*x1 + ((-4) + x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tl.where(tmp4, tmp14, tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/z5/cz5up7nzhq325byocbbjxbds535tutc2qn7yfou4jr4tmgsrepjo.py
# Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat_2 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_1, %permute_28], 2), kwargs = {})
triton_poi_fused_cat_13 = async_compile.triton('triton_poi_fused_cat_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 8)
    x3 = xindex // 8
    x1 = ((xindex // 8) % 4)
    x2 = xindex // 32
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x3 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (4*x1 + 64*x2 + ((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr1 + (16 + 4*x1 + 64*x2 + ((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.load(in_ptr1 + (32 + 4*x1 + 64*x2 + ((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.load(in_ptr1 + (48 + 4*x1 + 64*x2 + ((-4) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tmp13 + tmp14
    tmp16 = 4.0
    tmp17 = tmp15 / tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp6, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp5, tmp19)
    tl.store(out_ptr0 + (x4), tmp20, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/t6/ct6m2el4uo2eb56sa77ygnaver7chqohljll2qswcpot4wxq45wk.py
# Topologically Sorted Source Nodes: [combining_weights, sub_1, mul_2, mul_3, X_tilde_3], Original ATen: [aten.sigmoid, aten.rsub, aten.mul, aten.add]
# Source node to ATen node mapping:
#   X_tilde_3 => add_15
#   combining_weights => sigmoid
#   mul_2 => mul_10
#   mul_3 => mul_11
#   sub_1 => sub_7
# Graph fragment:
#   %sigmoid : [num_users=3] = call_function[target=torch.ops.aten.sigmoid.default](args = (%view_55,), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (1, %sigmoid), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %view_53), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sigmoid, %view_25), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_10, %mul_11), kwargs = {})
triton_poi_fused_add_mul_rsub_sigmoid_14 = async_compile.triton('triton_poi_fused_add_mul_rsub_sigmoid_14', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_rsub_sigmoid_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_rsub_sigmoid_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr1 + (x2), xmask)
    tmp8 = tl.load(in_ptr2 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = 1.0
    tmp5 = tmp4 - tmp3
    tmp7 = tmp5 * tmp6
    tmp9 = tmp3 * tmp8
    tmp10 = tmp7 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_3, (4, 8), (8, 1))
    assert_size_stride(primals_4, (4, ), (1, ))
    assert_size_stride(primals_5, (1, 4, 4), (16, 4, 1))
    assert_size_stride(primals_6, (16, 4), (4, 1))
    assert_size_stride(primals_7, (16, 4), (4, 1))
    assert_size_stride(primals_8, (16, 4), (4, 1))
    assert_size_stride(primals_9, (4, 16), (16, 1))
    assert_size_stride(primals_10, (4, ), (1, ))
    assert_size_stride(primals_11, (4, ), (1, ))
    assert_size_stride(primals_12, (4, 4), (4, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (4, 4), (4, 1))
    assert_size_stride(primals_15, (4, ), (1, ))
    assert_size_stride(primals_16, (4, ), (1, ))
    assert_size_stride(primals_17, (4, ), (1, ))
    assert_size_stride(primals_18, (4, 4), (4, 1))
    assert_size_stride(primals_19, (4, ), (1, ))
    assert_size_stride(primals_20, (4, 8), (8, 1))
    assert_size_stride(primals_21, (4, ), (1, ))
    assert_size_stride(primals_22, (1, 4, 4), (16, 4, 1))
    assert_size_stride(primals_23, (16, 4), (4, 1))
    assert_size_stride(primals_24, (16, 4), (4, 1))
    assert_size_stride(primals_25, (16, 4), (4, 1))
    assert_size_stride(primals_26, (4, 16), (16, 1))
    assert_size_stride(primals_27, (4, ), (1, ))
    assert_size_stride(primals_28, (4, ), (1, ))
    assert_size_stride(primals_29, (4, 4), (4, 1))
    assert_size_stride(primals_30, (4, ), (1, ))
    assert_size_stride(primals_31, (4, 4), (4, 1))
    assert_size_stride(primals_32, (4, ), (1, ))
    assert_size_stride(primals_33, (4, ), (1, ))
    assert_size_stride(primals_34, (4, ), (1, ))
    assert_size_stride(primals_35, (4, 4), (4, 1))
    assert_size_stride(primals_36, (4, ), (1, ))
    assert_size_stride(primals_37, (4, 4), (4, 1))
    assert_size_stride(primals_38, (4, ), (1, ))
    assert_size_stride(primals_39, (4, 8), (8, 1))
    assert_size_stride(primals_40, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 8), (32, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [X], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_2, primals_1, buf0, 128, grid=grid(128), stream=stream0)
        buf1 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [X_embedding], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (16, 8), (8, 1), 0), reinterpret_tensor(primals_3, (8, 4), (1, 8), 0), out=buf1)
        del primals_3
        buf2 = reinterpret_tensor(buf1, (4, 4, 4), (16, 4, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_with_pos], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf2, primals_4, primals_5, 64, grid=grid(64), stream=stream0)
        del primals_4
        del primals_5
        buf3 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (16, 4), (4, 1), 0), reinterpret_tensor(primals_6, (4, 16), (1, 4), 0), out=buf3)
        buf4 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (16, 4), (4, 1), 0), reinterpret_tensor(primals_7, (4, 16), (1, 4), 0), out=buf4)
        buf5 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (16, 4), (4, 1), 0), reinterpret_tensor(primals_8, (4, 16), (1, 4), 0), out=buf5)
        buf6 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv, attn], Original ATen: [aten.div, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_div_2.run(buf3, buf6, 256, grid=grid(256), stream=stream0)
        buf7 = reinterpret_tensor(buf3, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf3  # reuse
        # Topologically Sorted Source Nodes: [attn], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf4, buf7, 64, 4, grid=grid(64, 4), stream=stream0)
        buf8 = reinterpret_tensor(buf4, (16, 4, 4), (16, 4, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [attn], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf6, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf7, (16, 4, 4), (16, 4, 1), 0), out=buf8)
        buf9 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf8, buf9, 256, grid=grid(256), stream=stream0)
        buf10 = reinterpret_tensor(buf8, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf8  # reuse
        # Topologically Sorted Source Nodes: [attn_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf9, buf10, 256, grid=grid(256), stream=stream0)
        buf11 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf5, buf11, 256, grid=grid(256), stream=stream0)
        buf12 = reinterpret_tensor(buf5, (16, 4, 4), (16, 4, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf10, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf11, (16, 4, 4), (16, 4, 1), 0), out=buf12)
        buf13 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf12, buf13, 256, grid=grid(256), stream=stream0)
        buf14 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf13, (16, 16), (16, 1), 0), reinterpret_tensor(primals_9, (16, 4), (1, 16), 0), out=buf14)
        buf15 = reinterpret_tensor(buf14, (4, 4, 4), (16, 4, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [enc_output_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_7.run(buf15, buf2, 64, grid=grid(64), stream=stream0)
        buf16 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        buf17 = empty_strided_cuda((4, 4, 1), (4, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [enc_output_2], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_8.run(buf15, buf16, buf17, 16, grid=grid(16), stream=stream0)
        buf18 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [enc_output_2], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_9.run(buf15, buf16, buf17, primals_10, primals_11, buf18, 64, grid=grid(64), stream=stream0)
        del primals_11
        buf19 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf18, (16, 4), (4, 1), 0), reinterpret_tensor(primals_12, (4, 4), (1, 4), 0), out=buf19)
        buf20 = reinterpret_tensor(buf19, (4, 4, 4), (16, 4, 1), 0); del buf19  # reuse
        buf62 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_10.run(buf20, primals_13, buf62, 64, grid=grid(64), stream=stream0)
        del primals_13
        buf21 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf20, (16, 4), (4, 1), 0), reinterpret_tensor(primals_14, (4, 4), (1, 4), 0), out=buf21)
        buf22 = reinterpret_tensor(buf21, (4, 4, 4), (16, 4, 1), 0); del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_11.run(buf22, primals_15, buf18, 64, grid=grid(64), stream=stream0)
        del primals_15
        buf23 = buf17; del buf17  # reuse
        buf24 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_8.run(buf22, buf23, buf24, 16, grid=grid(16), stream=stream0)
        buf25 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_9.run(buf22, buf23, buf24, primals_16, primals_17, buf25, 64, grid=grid(64), stream=stream0)
        del primals_17
        buf26 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [X_tilde_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_19, reinterpret_tensor(buf25, (16, 4), (4, 1), 0), reinterpret_tensor(primals_18, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf26)
        del primals_19
        buf27 = empty_strided_cuda((4, 4, 8), (32, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [X_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_12.run(primals_1, primals_2, buf26, buf27, 128, grid=grid(128), stream=stream0)
        del primals_2
        buf28 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [X_embedding_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf27, (16, 8), (8, 1), 0), reinterpret_tensor(primals_20, (8, 4), (1, 8), 0), out=buf28)
        buf29 = reinterpret_tensor(buf28, (4, 4, 4), (16, 4, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [x_with_pos_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_1.run(buf29, primals_21, primals_22, 64, grid=grid(64), stream=stream0)
        del primals_21
        del primals_22
        buf30 = reinterpret_tensor(buf12, (16, 16), (16, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (16, 4), (4, 1), 0), reinterpret_tensor(primals_23, (4, 16), (1, 4), 0), out=buf30)
        buf31 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (16, 4), (4, 1), 0), reinterpret_tensor(primals_24, (4, 16), (1, 4), 0), out=buf31)
        buf32 = empty_strided_cuda((16, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (16, 4), (4, 1), 0), reinterpret_tensor(primals_25, (4, 16), (1, 4), 0), out=buf32)
        buf33 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [truediv_1, attn_3], Original ATen: [aten.div, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_div_2.run(buf30, buf33, 256, grid=grid(256), stream=stream0)
        buf34 = reinterpret_tensor(buf30, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [attn_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_3.run(buf31, buf34, 64, 4, grid=grid(64, 4), stream=stream0)
        buf35 = reinterpret_tensor(buf31, (16, 4, 4), (16, 4, 1), 0); del buf31  # reuse
        # Topologically Sorted Source Nodes: [attn_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf33, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf34, (16, 4, 4), (16, 4, 1), 0), out=buf35)
        buf36 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attn_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_4.run(buf35, buf36, 256, grid=grid(256), stream=stream0)
        buf37 = reinterpret_tensor(buf35, (4, 4, 4, 4), (64, 16, 4, 1), 0); del buf35  # reuse
        # Topologically Sorted Source Nodes: [attn_4], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_5.run(buf36, buf37, 256, grid=grid(256), stream=stream0)
        buf38 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf32, buf38, 256, grid=grid(256), stream=stream0)
        buf39 = reinterpret_tensor(buf32, (16, 4, 4), (16, 4, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [output_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf37, (16, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf38, (16, 4, 4), (16, 4, 1), 0), out=buf39)
        buf40 = empty_strided_cuda((4, 4, 4, 4), (64, 16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [contiguous_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf39, buf40, 256, grid=grid(256), stream=stream0)
        del buf39
        buf41 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [v_7], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf40, (16, 16), (16, 1), 0), reinterpret_tensor(primals_26, (16, 4), (1, 16), 0), out=buf41)
        buf42 = reinterpret_tensor(buf41, (4, 4, 4), (16, 4, 1), 0); del buf41  # reuse
        # Topologically Sorted Source Nodes: [enc_output_4], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_7.run(buf42, buf29, 64, grid=grid(64), stream=stream0)
        buf43 = buf24; del buf24  # reuse
        buf44 = buf23; del buf23  # reuse
        # Topologically Sorted Source Nodes: [enc_output_5], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_8.run(buf42, buf43, buf44, 16, grid=grid(16), stream=stream0)
        buf45 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [enc_output_5], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_9.run(buf42, buf43, buf44, primals_27, primals_28, buf45, 64, grid=grid(64), stream=stream0)
        del primals_28
        buf46 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf45, (16, 4), (4, 1), 0), reinterpret_tensor(primals_29, (4, 4), (1, 4), 0), out=buf46)
        buf47 = reinterpret_tensor(buf46, (4, 4, 4), (16, 4, 1), 0); del buf46  # reuse
        buf61 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [x_5], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_10.run(buf47, primals_30, buf61, 64, grid=grid(64), stream=stream0)
        del primals_30
        buf48 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf47, (16, 4), (4, 1), 0), reinterpret_tensor(primals_31, (4, 4), (1, 4), 0), out=buf48)
        buf49 = reinterpret_tensor(buf48, (4, 4, 4), (16, 4, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_11.run(buf49, primals_32, buf45, 64, grid=grid(64), stream=stream0)
        del primals_32
        buf50 = buf44; del buf44  # reuse
        buf51 = buf43; del buf43  # reuse
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_8.run(buf49, buf50, buf51, 16, grid=grid(16), stream=stream0)
        buf52 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_native_layer_norm_9.run(buf49, buf50, buf51, primals_33, primals_34, buf52, 64, grid=grid(64), stream=stream0)
        del buf50
        del buf51
        del primals_34
        buf53 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf52, (16, 4), (4, 1), 0), reinterpret_tensor(primals_35, (4, 4), (1, 4), 0), out=buf53)
        buf54 = reinterpret_tensor(buf53, (4, 4, 4), (16, 4, 1), 0); del buf53  # reuse
        buf60 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_10.run(buf54, primals_36, buf60, 64, grid=grid(64), stream=stream0)
        del primals_36
        buf55 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [X_tilde_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_38, reinterpret_tensor(buf54, (16, 4), (4, 1), 0), reinterpret_tensor(primals_37, (4, 4), (1, 4), 0), alpha=1, beta=1, out=buf55)
        del primals_38
        buf56 = empty_strided_cuda((4, 4, 8), (32, 8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_13.run(primals_1, buf37, buf56, 128, grid=grid(128), stream=stream0)
        buf57 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf56, (16, 8), (8, 1), 0), reinterpret_tensor(primals_39, (8, 4), (1, 8), 0), out=buf57)
        buf58 = reinterpret_tensor(buf57, (4, 4, 4), (16, 4, 1), 0); del buf57  # reuse
        buf59 = empty_strided_cuda((4, 4, 4), (16, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [combining_weights, sub_1, mul_2, mul_3, X_tilde_3], Original ATen: [aten.sigmoid, aten.rsub, aten.mul, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_rsub_sigmoid_14.run(buf58, primals_40, buf55, buf26, buf59, 64, grid=grid(64), stream=stream0)
        del primals_40
    return (reinterpret_tensor(buf26, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf55, (4, 4, 4), (16, 4, 1), 0), buf59, buf10, buf37, buf58, primals_1, primals_10, primals_16, primals_27, primals_33, reinterpret_tensor(buf0, (16, 8), (8, 1), 0), reinterpret_tensor(buf2, (16, 4), (4, 1), 0), buf10, reinterpret_tensor(buf13, (16, 16), (16, 1), 0), buf15, reinterpret_tensor(buf18, (16, 4), (4, 1), 0), reinterpret_tensor(buf20, (16, 4), (4, 1), 0), buf22, reinterpret_tensor(buf25, (16, 4), (4, 1), 0), reinterpret_tensor(buf26, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf27, (16, 8), (8, 1), 0), reinterpret_tensor(buf29, (16, 4), (4, 1), 0), buf37, reinterpret_tensor(buf40, (16, 16), (16, 1), 0), buf42, reinterpret_tensor(buf45, (16, 4), (4, 1), 0), reinterpret_tensor(buf47, (16, 4), (4, 1), 0), buf49, reinterpret_tensor(buf52, (16, 4), (4, 1), 0), reinterpret_tensor(buf54, (16, 4), (4, 1), 0), reinterpret_tensor(buf55, (4, 4, 4), (16, 4, 1), 0), reinterpret_tensor(buf56, (16, 8), (8, 1), 0), buf58, primals_39, primals_37, buf60, primals_35, primals_31, buf61, primals_29, primals_26, reinterpret_tensor(buf38, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf33, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf34, (16, 4, 4), (16, 1, 4), 0), primals_25, primals_24, primals_23, primals_20, primals_18, primals_14, buf62, primals_12, primals_9, reinterpret_tensor(buf11, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf6, (16, 4, 4), (16, 1, 4), 0), reinterpret_tensor(buf7, (16, 4, 4), (16, 1, 4), 0), primals_8, primals_7, primals_6, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((1, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((16, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((4, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
