# AOT ID: ['25_forward']
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


# kernel path: inductor_cache/pv/cpvme26tpuy662q37ccvyrmnzr3pas3fh3tnibj6n5uuild2t2z6.py
# Topologically Sorted Source Nodes: [pos_encoding_2, embeddings_2], Original ATen: [aten.repeat, aten.add]
# Source node to ATen node mapping:
#   embeddings_2 => add_2
#   pos_encoding_2 => repeat
# Graph fragment:
#   %repeat : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%unsqueeze_2, [1, 4, 1]), kwargs = {})
#   %add_2 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute, %repeat), kwargs = {})
triton_poi_fused_add_repeat_0 = async_compile.triton('triton_poi_fused_add_repeat_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_repeat_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_repeat_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 36) % 128)
    x0 = (xindex % 36)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = x1
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tmp3 >= tmp4
    tmp6 = tl.full([1], 64, tl.int64)
    tmp7 = tmp3 < tmp6
    tmp8 = 2*(x1)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = -0.07195578515529633
    tmp11 = tmp9 * tmp10
    tmp12 = tl_math.exp(tmp11)
    tmp13 = x0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp12
    tmp16 = tl_math.sin(tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp7, tmp16, tmp17)
    tmp19 = tmp3 >= tmp6
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp3 < tmp20
    tmp22 = 2*((-64) + x1)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = -0.07195578515529633
    tmp25 = tmp23 * tmp24
    tmp26 = tl_math.exp(tmp25)
    tmp27 = x0
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp26
    tmp30 = tl_math.cos(tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp19, tmp30, tmp31)
    tmp33 = tl.where(tmp7, tmp18, tmp32)
    tmp34 = tmp2 + tmp33
    tl.store(in_out_ptr0 + (x3), tmp34, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/gd/cgdasbcutir3qfbsfemsvy4tu26rqwlp2p7nqgbautv4d6xcvhpr.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone
# Graph fragment:
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%add_2,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 512}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 36*x1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + 512*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/yi/cyib7qvgqh3wjrgpk4lnui5vkzl5hrwpgvxvol75hqf4lruxjhh6.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_1
# Graph fragment:
#   %clone_1 : [num_users=3] = call_function[target=torch.ops.aten.clone.default](args = (%squeeze,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 65536}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 55296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 144)
    x2 = xindex // 18432
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 384*x1), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + 128*x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/wb/cwbr6674x7k7y7xjni26rclcwatpirn3dptey36mozbyhqfmfs2u.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_7
# Graph fragment:
#   %view_7 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_3, [4, 4, 36, 32]), kwargs = {})
triton_poi_fused_view_3 = async_compile.triton('triton_poi_fused_view_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tk/ctkekslagh5l5nhtspuzb6qzdupe3kh7mlzjr2bfyd63dbchzkzf.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_8
# Graph fragment:
#   %view_8 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_4, [4, 4, 36, 32]), kwargs = {})
triton_poi_fused_view_4 = async_compile.triton('triton_poi_fused_view_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (18432 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ub/cubwhitk2j4ubcvi6tem5xsaamd4b63ywm4llb5qg2ahdgh5hhd2.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   multi_head_attention_forward => view_9
# Graph fragment:
#   %view_9 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%permute_5, [4, 4, 36, 32]), kwargs = {})
triton_poi_fused_view_5 = async_compile.triton('triton_poi_fused_view_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (36864 + x4), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/pq/cpqx3ulkxisxixvctamxtyhadr3j3nuw7zb26heahopahd7khpou.py
# Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   multi_head_attention_forward => clone_2
# Graph fragment:
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_6 = async_compile.triton('triton_poi_fused_clone_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 4)
    x2 = xindex // 512
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128*x2 + 4608*x1), xmask)
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/oc/coccbcyamyg66u6dgjnz3zyre63bma7ihima5z3yuynmmle2xb5z.py
# Topologically Sorted Source Nodes: [add_1, x], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_1 => add_4
#   x => add_5, clone_4, rsqrt, var_mean
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_11), kwargs = {})
#   %clone_4 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_4,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 128), kwargs = {})
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7 = async_compile.triton('triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 2, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x1 + 36*r2 + 4608*x0), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + 128*x3), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
    tl.store(out_ptr1 + (x3), tmp7, xmask)
    tmp9 = 128.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.rsqrt(tmp12)
    tmp14 = 0.0078125
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr2 + (x3), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ue/cuezwpyzmrfzsvk3mdk2ecowuynelffkh4aenby4vtokvwqdcunh.py
# Topologically Sorted Source Nodes: [add_1, x], Original ATen: [aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_1 => add_4
#   x => add_5, add_6, clone_4, mul_4, mul_5, rsqrt, sub, var_mean
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %view_11), kwargs = {})
#   %clone_4 : [num_users=2] = call_function[target=torch.ops.aten.clone.default](args = (%add_4,), kwargs = {memory_format: torch.contiguous_format})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%clone_4, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_5,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%clone_4, %getitem_5), kwargs = {})
#   %mul_4 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %primals_8), kwargs = {})
#   %add_6 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %primals_9), kwargs = {})
triton_poi_fused_add_native_layer_norm_8 = async_compile.triton('triton_poi_fused_add_native_layer_norm_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 64, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_layer_norm_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex
    x1 = (xindex % 128)
    x2 = xindex // 128
    tmp0 = tl.load(in_ptr0 + (y0 + 36*x3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + 512*y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x2 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x2 + 4*y0), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 128.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + 512*y0), tmp13, xmask & ymask)
    tl.store(out_ptr0 + (x3 + 512*y0), tmp17, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/tt/ctthte2eqxond7jowuobkahm4zjic34fhc45dqjrhtszwzumauk2.py
# Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
# Source node to ATen node mapping:
#   relu => relu
# Graph fragment:
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%view_13,), kwargs = {})
#   %le_3 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%relu, 0), kwargs = {})
triton_poi_fused_relu_threshold_backward_9 = async_compile.triton('triton_poi_fused_relu_threshold_backward_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_threshold_backward_9(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 147456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp5 = 0.0
    tmp6 = tmp4 <= tmp5
    tl.store(in_out_ptr0 + (x2), tmp4, None)
    tl.store(out_ptr0 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/ii/ciiccb2fxvswssixbphpjmpxa7k24toswtmiu6drmnguqajjqnoc.py
# Topologically Sorted Source Nodes: [add_2, x_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   add_2 => add_7
#   x_2 => add_8, add_9, mul_6, mul_7, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_15), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_7, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-05), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_8,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_7, %getitem_7), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_14), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_15), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_1, 128), kwargs = {})
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10 = async_compile.triton('triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 256, 'r': 128},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 144
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp1 = tl.load(in_out_ptr0 + (r1 + 128*x0), xmask, other=0.0)
    tmp2 = tl.load(in_ptr1 + (r1), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (r1), None, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr3 + (r1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 128.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = 0.0078125
    tmp33 = tmp26 * tmp32
    tl.store(in_out_ptr0 + (r1 + 128*x0), tmp27, xmask)
    tl.store(out_ptr2 + (r1 + 128*x0), tmp31, xmask)
    tl.store(out_ptr3 + (x0), tmp33, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51 = args
    args.clear()
    assert_size_stride(primals_1, (128, 4, 10, 10), (400, 100, 10, 1))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (4, 4, 64, 64), (16384, 4096, 64, 1))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_5, (384, 128), (128, 1))
    assert_size_stride(primals_6, (128, 128), (128, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (1024, 128), (128, 1))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (128, 1024), (1024, 1))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (384, ), (1, ))
    assert_size_stride(primals_17, (384, 128), (128, 1))
    assert_size_stride(primals_18, (128, 128), (128, 1))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (1024, 128), (128, 1))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (128, 1024), (1024, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (384, ), (1, ))
    assert_size_stride(primals_29, (384, 128), (128, 1))
    assert_size_stride(primals_30, (128, 128), (128, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (1024, 128), (128, 1))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (128, 1024), (1024, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, 128), (128, 1))
    assert_size_stride(primals_42, (128, 128), (128, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (1024, 128), (128, 1))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_48, (128, 1024), (1024, 1))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_3, primals_1, stride=(10, 10), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 128, 6, 6), (4608, 36, 6, 1))
        buf2 = reinterpret_tensor(buf0, (36, 4, 128), (1, 4608, 36), 0); del buf0  # reuse
        # Topologically Sorted Source Nodes: [pos_encoding_2, embeddings_2], Original ATen: [aten.repeat, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_repeat_0.run(buf2, primals_2, 18432, grid=grid(18432), stream=stream0)
        del primals_2
        buf3 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_1.run(buf2, buf3, 36, 512, grid=grid(36, 512), stream=stream0)
        buf4 = empty_strided_cuda((144, 384), (384, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (144, 128), (128, 1), 0), reinterpret_tensor(primals_5, (128, 384), (1, 128), 0), out=buf4)
        buf5 = empty_strided_cuda((3, 36, 4, 128), (18432, 512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf4, primals_4, buf5, 55296, grid=grid(55296), stream=stream0)
        del primals_4
        buf6 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf5, buf6, 18432, grid=grid(18432), stream=stream0)
        buf7 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf5, buf7, 18432, grid=grid(18432), stream=stream0)
        buf8 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_5.run(buf5, buf8, 18432, grid=grid(18432), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf6, buf7, buf8, None, True)
        buf10 = buf9[0]
        buf11 = buf9[1]
        buf12 = buf9[2]
        buf13 = buf9[3]
        del buf9
        buf14 = empty_strided_cuda((36, 4, 4, 32), (512, 128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf10, buf14, 18432, grid=grid(18432), stream=stream0)
        buf15 = empty_strided_cuda((144, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf14, (144, 128), (128, 1), 0), reinterpret_tensor(primals_6, (128, 128), (1, 128), 0), out=buf15)
        buf16 = empty_strided_cuda((36, 4, 1), (4, 1, 144), torch.float32)
        buf17 = empty_strided_cuda((36, 4, 1), (4, 1, 144), torch.float32)
        buf115 = empty_strided_cuda((36, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_1, x], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_7.run(buf2, buf15, primals_7, buf16, buf17, buf115, 144, 128, grid=grid(144), stream=stream0)
        buf19 = reinterpret_tensor(buf15, (36, 4, 128), (512, 128, 1), 0); del buf15  # reuse
        buf20 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_1, x], Original ATen: [aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_layer_norm_8.run(buf19, buf2, primals_7, buf16, buf17, primals_8, primals_9, buf20, 36, 512, grid=grid(36, 512), stream=stream0)
        del primals_7
        del primals_9
        buf21 = empty_strided_cuda((144, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf20, (144, 128), (128, 1), 0), reinterpret_tensor(primals_10, (128, 1024), (1, 128), 0), out=buf21)
        buf22 = reinterpret_tensor(buf21, (36, 4, 1024), (4096, 1024, 1), 0); del buf21  # reuse
        buf114 = empty_strided_cuda((36, 4, 1024), (4096, 1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9.run(buf22, primals_11, buf114, 147456, grid=grid(147456), stream=stream0)
        del primals_11
        buf23 = reinterpret_tensor(buf2, (144, 128), (128, 1), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf22, (144, 1024), (1024, 1), 0), reinterpret_tensor(primals_12, (1024, 128), (1, 1024), 0), out=buf23)
        buf27 = reinterpret_tensor(buf23, (36, 4, 128), (512, 128, 1), 0); del buf23  # reuse
        buf28 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        buf113 = reinterpret_tensor(buf17, (36, 4, 1), (4, 1, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [add_2, x_2], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf27, buf20, primals_13, primals_14, primals_15, buf28, buf113, 144, 128, grid=grid(144), stream=stream0)
        del primals_13
        del primals_15
        buf29 = reinterpret_tensor(buf5, (144, 384), (384, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf28, (144, 128), (128, 1), 0), reinterpret_tensor(primals_17, (128, 384), (1, 128), 0), out=buf29)
        buf30 = reinterpret_tensor(buf4, (3, 36, 4, 128), (18432, 512, 128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf29, primals_16, buf30, 55296, grid=grid(55296), stream=stream0)
        del primals_16
        buf31 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf30, buf31, 18432, grid=grid(18432), stream=stream0)
        buf32 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf30, buf32, 18432, grid=grid(18432), stream=stream0)
        buf33 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_5.run(buf30, buf33, 18432, grid=grid(18432), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf34 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf31, buf32, buf33, None, True)
        buf35 = buf34[0]
        buf36 = buf34[1]
        buf37 = buf34[2]
        buf38 = buf34[3]
        del buf34
        buf39 = empty_strided_cuda((36, 4, 4, 32), (512, 128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf35, buf39, 18432, grid=grid(18432), stream=stream0)
        buf40 = empty_strided_cuda((144, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf39, (144, 128), (128, 1), 0), reinterpret_tensor(primals_18, (128, 128), (1, 128), 0), out=buf40)
        buf44 = reinterpret_tensor(buf40, (36, 4, 128), (512, 128, 1), 0); del buf40  # reuse
        buf45 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        buf112 = reinterpret_tensor(buf16, (36, 4, 1), (4, 1, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [add_3, x_3], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf44, buf28, primals_19, primals_20, primals_21, buf45, buf112, 144, 128, grid=grid(144), stream=stream0)
        del primals_19
        del primals_21
        buf46 = empty_strided_cuda((144, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf45, (144, 128), (128, 1), 0), reinterpret_tensor(primals_22, (128, 1024), (1, 128), 0), out=buf46)
        buf47 = reinterpret_tensor(buf46, (36, 4, 1024), (4096, 1024, 1), 0); del buf46  # reuse
        buf111 = empty_strided_cuda((36, 4, 1024), (4096, 1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_1], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9.run(buf47, primals_23, buf111, 147456, grid=grid(147456), stream=stream0)
        del primals_23
        buf48 = empty_strided_cuda((144, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf47, (144, 1024), (1024, 1), 0), reinterpret_tensor(primals_24, (1024, 128), (1, 1024), 0), out=buf48)
        buf52 = reinterpret_tensor(buf48, (36, 4, 128), (512, 128, 1), 0); del buf48  # reuse
        buf53 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        buf110 = empty_strided_cuda((36, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_4, x_5], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf52, buf45, primals_25, primals_26, primals_27, buf53, buf110, 144, 128, grid=grid(144), stream=stream0)
        del primals_25
        del primals_27
        buf54 = reinterpret_tensor(buf30, (144, 384), (384, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf53, (144, 128), (128, 1), 0), reinterpret_tensor(primals_29, (128, 384), (1, 128), 0), out=buf54)
        buf55 = reinterpret_tensor(buf29, (3, 36, 4, 128), (18432, 512, 128, 1), 0); del buf29  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf54, primals_28, buf55, 55296, grid=grid(55296), stream=stream0)
        del primals_28
        buf56 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf55, buf56, 18432, grid=grid(18432), stream=stream0)
        buf57 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf55, buf57, 18432, grid=grid(18432), stream=stream0)
        buf58 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_5.run(buf55, buf58, 18432, grid=grid(18432), stream=stream0)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf59 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf56, buf57, buf58, None, True)
        buf60 = buf59[0]
        buf61 = buf59[1]
        buf62 = buf59[2]
        buf63 = buf59[3]
        del buf59
        buf64 = empty_strided_cuda((36, 4, 4, 32), (512, 128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf60, buf64, 18432, grid=grid(18432), stream=stream0)
        buf65 = empty_strided_cuda((144, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf64, (144, 128), (128, 1), 0), reinterpret_tensor(primals_30, (128, 128), (1, 128), 0), out=buf65)
        buf69 = reinterpret_tensor(buf65, (36, 4, 128), (512, 128, 1), 0); del buf65  # reuse
        buf70 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        buf109 = empty_strided_cuda((36, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_5, x_6], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf69, buf53, primals_31, primals_32, primals_33, buf70, buf109, 144, 128, grid=grid(144), stream=stream0)
        del primals_31
        del primals_33
        buf71 = empty_strided_cuda((144, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf70, (144, 128), (128, 1), 0), reinterpret_tensor(primals_34, (128, 1024), (1, 128), 0), out=buf71)
        buf72 = reinterpret_tensor(buf71, (36, 4, 1024), (4096, 1024, 1), 0); del buf71  # reuse
        buf108 = empty_strided_cuda((36, 4, 1024), (4096, 1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_2], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9.run(buf72, primals_35, buf108, 147456, grid=grid(147456), stream=stream0)
        del primals_35
        buf73 = empty_strided_cuda((144, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf72, (144, 1024), (1024, 1), 0), reinterpret_tensor(primals_36, (1024, 128), (1, 1024), 0), out=buf73)
        buf77 = reinterpret_tensor(buf73, (36, 4, 128), (512, 128, 1), 0); del buf73  # reuse
        buf78 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        buf107 = empty_strided_cuda((36, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_6, x_8], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf77, buf70, primals_37, primals_38, primals_39, buf78, buf107, 144, 128, grid=grid(144), stream=stream0)
        del primals_37
        del primals_39
        buf79 = reinterpret_tensor(buf55, (144, 384), (384, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf78, (144, 128), (128, 1), 0), reinterpret_tensor(primals_41, (128, 384), (1, 128), 0), out=buf79)
        buf80 = reinterpret_tensor(buf54, (3, 36, 4, 128), (18432, 512, 128, 1), 0); del buf54  # reuse
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_2.run(buf79, primals_40, buf80, 55296, grid=grid(55296), stream=stream0)
        del buf79
        del primals_40
        buf81 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_3.run(buf80, buf81, 18432, grid=grid(18432), stream=stream0)
        buf82 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_4.run(buf80, buf82, 18432, grid=grid(18432), stream=stream0)
        buf83 = empty_strided_cuda((4, 4, 36, 32), (128, 32, 512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_5.run(buf80, buf83, 18432, grid=grid(18432), stream=stream0)
        del buf80
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf84 = torch.ops.aten._scaled_dot_product_efficient_attention.default(buf81, buf82, buf83, None, True)
        buf85 = buf84[0]
        buf86 = buf84[1]
        buf87 = buf84[2]
        buf88 = buf84[3]
        del buf84
        buf89 = empty_strided_cuda((36, 4, 4, 32), (512, 128, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_6.run(buf85, buf89, 18432, grid=grid(18432), stream=stream0)
        buf90 = empty_strided_cuda((144, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [multi_head_attention_forward_3], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf89, (144, 128), (128, 1), 0), reinterpret_tensor(primals_42, (128, 128), (1, 128), 0), out=buf90)
        buf94 = reinterpret_tensor(buf90, (36, 4, 128), (512, 128, 1), 0); del buf90  # reuse
        buf95 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        buf106 = empty_strided_cuda((36, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_7, x_9], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf94, buf78, primals_43, primals_44, primals_45, buf95, buf106, 144, 128, grid=grid(144), stream=stream0)
        del primals_43
        del primals_45
        buf96 = empty_strided_cuda((144, 1024), (1024, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf95, (144, 128), (128, 1), 0), reinterpret_tensor(primals_46, (128, 1024), (1, 128), 0), out=buf96)
        buf97 = reinterpret_tensor(buf96, (36, 4, 1024), (4096, 1024, 1), 0); del buf96  # reuse
        buf105 = empty_strided_cuda((36, 4, 1024), (4096, 1024, 1), torch.bool)
        # Topologically Sorted Source Nodes: [relu_3], Original ATen: [aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_threshold_backward_9.run(buf97, primals_47, buf105, 147456, grid=grid(147456), stream=stream0)
        del primals_47
        buf98 = empty_strided_cuda((144, 128), (128, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf97, (144, 1024), (1024, 1), 0), reinterpret_tensor(primals_48, (1024, 128), (1, 1024), 0), out=buf98)
        buf102 = reinterpret_tensor(buf98, (36, 4, 128), (512, 128, 1), 0); del buf98  # reuse
        buf103 = empty_strided_cuda((36, 4, 128), (512, 128, 1), torch.float32)
        buf104 = empty_strided_cuda((36, 4, 1), (4, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [add_8, x_11], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_10.run(buf102, buf95, primals_49, primals_50, primals_51, buf103, buf104, 144, 128, grid=grid(144), stream=stream0)
        del primals_49
        del primals_51
    return (buf103, primals_1, primals_3, primals_8, primals_14, primals_20, primals_26, primals_32, primals_38, primals_44, primals_50, reinterpret_tensor(buf3, (144, 128), (128, 1), 0), buf6, buf7, buf8, buf10, buf11, buf12, buf13, reinterpret_tensor(buf14, (144, 128), (128, 1), 0), buf19, reinterpret_tensor(buf20, (144, 128), (128, 1), 0), reinterpret_tensor(buf22, (144, 1024), (1024, 1), 0), buf27, reinterpret_tensor(buf28, (144, 128), (128, 1), 0), buf31, buf32, buf33, buf35, buf36, buf37, buf38, reinterpret_tensor(buf39, (144, 128), (128, 1), 0), buf44, reinterpret_tensor(buf45, (144, 128), (128, 1), 0), reinterpret_tensor(buf47, (144, 1024), (1024, 1), 0), buf52, reinterpret_tensor(buf53, (144, 128), (128, 1), 0), buf56, buf57, buf58, buf60, buf61, buf62, buf63, reinterpret_tensor(buf64, (144, 128), (128, 1), 0), buf69, reinterpret_tensor(buf70, (144, 128), (128, 1), 0), reinterpret_tensor(buf72, (144, 1024), (1024, 1), 0), buf77, reinterpret_tensor(buf78, (144, 128), (128, 1), 0), buf81, buf82, buf83, buf85, buf86, buf87, buf88, reinterpret_tensor(buf89, (144, 128), (128, 1), 0), buf94, reinterpret_tensor(buf95, (144, 128), (128, 1), 0), reinterpret_tensor(buf97, (144, 1024), (1024, 1), 0), buf102, buf104, primals_48, buf105, primals_46, buf106, primals_42, primals_41, buf107, primals_36, buf108, primals_34, buf109, primals_30, primals_29, buf110, primals_24, buf111, primals_22, buf112, primals_18, primals_17, buf113, primals_12, buf114, primals_10, buf115, primals_6, primals_5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 4, 10, 10), (400, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 4, 64, 64), (16384, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
