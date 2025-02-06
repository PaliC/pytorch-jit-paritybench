# AOT ID: ['2_forward']
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


# kernel path: inductor_cache/ja/cjax6rbkththhotjxchjit6raesjusu64hbqlonngvbxghydtk3b.py
# Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   cat => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%primals_1, %primals_2], 1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: inductor_cache/6k/c6khsuftaf35ix6q6f7ls4wwtuavm6hj5i6o3ln6rdc4hnzehe4u.py
# Topologically Sorted Source Nodes: [zeros_like], Original ATen: [aten.zeros_like]
# Source node to ATen node mapping:
#   zeros_like => full_default
# Graph fragment:
#   %full_default : [num_users=1] = call_function[target=torch.ops.aten.full.default](args = ([4, 8], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
triton_poi_fused_zeros_like_1 = async_compile.triton('triton_poi_fused_zeros_like_1', '''
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
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_zeros_like_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_zeros_like_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/mi/cmidq2smbdtpfruq5pnxabfzcxwhrdkntpkwp5idmr6ttsydoqk5.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %view_3), kwargs = {})
triton_poi_fused_add_2 = async_compile.triton('triton_poi_fused_add_2', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 8)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/et/cetd3xtdfsdjepmpwqkfceic2oipllzb7qb37d56fbhyln5gqen2.py
# Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid => sigmoid
# Graph fragment:
#   %sigmoid : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select_2,), kwargs = {})
triton_poi_fused_sigmoid_3 = async_compile.triton('triton_poi_fused_sigmoid_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sigmoid_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rx/crxfokxycwthmfr7a3qc22wzxzkeu23pjpsj3le74t554xfff3zl.py
# Topologically Sorted Source Nodes: [sigmoid_1], Original ATen: [aten.sigmoid]
# Source node to ATen node mapping:
#   sigmoid_1 => sigmoid_1
# Graph fragment:
#   %sigmoid_1 : [num_users=2] = call_function[target=torch.ops.aten.sigmoid.default](args = (%select_3,), kwargs = {})
triton_poi_fused_sigmoid_4 = async_compile.triton('triton_poi_fused_sigmoid_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sigmoid_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sigmoid_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.sigmoid(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/kl/cklcxnlxpuwwdygncdpscvk4xvu3bkrblosukuqbjjtphik4fccx.py
# Topologically Sorted Source Nodes: [tanh], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   tanh => tanh
# Graph fragment:
#   %tanh : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%select_10,), kwargs = {})
triton_poi_fused_tanh_5 = async_compile.triton('triton_poi_fused_tanh_5', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxgajhveyhortt266dr3zlcshq4p5jddmufhsighhp3b7omfb3fe.py
# Topologically Sorted Source Nodes: [tanh_1], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   tanh_1 => tanh_1
# Graph fragment:
#   %tanh_1 : [num_users=2] = call_function[target=torch.ops.aten.tanh.default](args = (%select_11,), kwargs = {})
triton_poi_fused_tanh_6 = async_compile.triton('triton_poi_fused_tanh_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crbblcdqbglxbpfkfzavlhvwdnpp4gm34wecefahhohexegjmmea.py
# Topologically Sorted Source Nodes: [h_new], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   h_new => add_3
# Graph fragment:
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_19, %view_21), kwargs = {})
triton_poi_fused_add_7 = async_compile.triton('triton_poi_fused_add_7', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_7(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4), (4, 1))
    assert_size_stride(primals_2, (4, 4), (4, 1))
    assert_size_stride(primals_3, (4, 8), (8, 1))
    assert_size_stride(primals_4, (1, 4), (4, 1))
    assert_size_stride(primals_5, (4, 8), (8, 1))
    assert_size_stride(primals_6, (1, 4), (4, 1))
    assert_size_stride(primals_7, (4, 8), (8, 1))
    assert_size_stride(primals_8, (1, 4), (4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cat], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(primals_1, primals_2, buf0, 32, grid=grid(32), stream=stream0)
        buf1 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [zeros_like], Original ATen: [aten.zeros_like]
        stream0 = get_raw_stream(0)
        triton_poi_fused_zeros_like_1.run(buf1, 32, grid=grid(32), stream=stream0)
        # Topologically Sorted Source Nodes: [cat, zeros_like, inp], Original ATen: [aten.cat, aten.zeros_like, aten.complex]
        buf2 = torch.ops.aten.complex.default(buf0, buf1)
        del buf0
        buf3 = buf2
        del buf2
        # Topologically Sorted Source Nodes: [getattr_1], Original ATen: [aten.permute]
        buf4 = torch.ops.aten.permute.default(primals_3, [1, 0])
        buf5 = buf4
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        buf6 = torch.ops.aten.mm.default(buf3, buf5)
        del buf4
        del buf5
        del primals_3
        buf7 = buf6
        del buf6
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        buf8 = torch.ops.aten.view.dtype(buf7, torch.float32)
        buf9 = buf8
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        buf10 = torch.ops.aten.view.dtype(primals_4, torch.float32)
        buf11 = buf10
        buf12 = reinterpret_tensor(buf1, (4, 4, 2), (8, 2, 1), 0); del buf1  # reuse
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf9, buf11, buf12, 32, grid=grid(32), stream=stream0)
        del buf10
        del buf11
        del buf7
        del buf8
        del buf9
        del primals_4
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        buf13 = torch.ops.aten.view.dtype(reinterpret_tensor(buf12, (4, 8), (8, 1), 0), torch.complex64)
        buf14 = buf13
        # Topologically Sorted Source Nodes: [getattr_2], Original ATen: [aten.view_as_real]
        buf15 = torch.ops.aten.view_as_real.default(buf14)
        buf16 = buf15
        buf17 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid], Original ATen: [aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_3.run(buf16, buf17, 16, grid=grid(16), stream=stream0)
        buf18 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_1], Original ATen: [aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_4.run(buf16, buf18, 16, grid=grid(16), stream=stream0)
        del buf13
        del buf14
        del buf15
        del buf16
        # Topologically Sorted Source Nodes: [r], Original ATen: [aten.complex]
        buf19 = torch.ops.aten.complex.default(buf17, buf18)
        buf20 = buf19
        del buf19
        # Topologically Sorted Source Nodes: [getattr_4], Original ATen: [aten.permute]
        buf21 = torch.ops.aten.permute.default(primals_5, [1, 0])
        buf22 = buf21
        # Topologically Sorted Source Nodes: [matmul_1], Original ATen: [aten.mm]
        buf23 = torch.ops.aten.mm.default(buf3, buf22)
        del buf21
        del buf22
        del primals_5
        buf24 = buf23
        del buf23
        # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
        buf25 = torch.ops.aten.view.dtype(buf24, torch.float32)
        buf26 = buf25
        # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
        buf27 = torch.ops.aten.view.dtype(primals_6, torch.float32)
        buf28 = buf27
        buf29 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf26, buf28, buf29, 32, grid=grid(32), stream=stream0)
        del buf24
        del buf25
        del buf26
        del buf27
        del buf28
        del primals_6
        # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
        buf30 = torch.ops.aten.view.dtype(reinterpret_tensor(buf29, (4, 8), (8, 1), 0), torch.complex64)
        buf31 = buf30
        # Topologically Sorted Source Nodes: [getattr_5], Original ATen: [aten.view_as_real]
        buf32 = torch.ops.aten.view_as_real.default(buf31)
        buf33 = buf32
        buf34 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_2], Original ATen: [aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_3.run(buf33, buf34, 16, grid=grid(16), stream=stream0)
        buf35 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sigmoid_3], Original ATen: [aten.sigmoid]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sigmoid_4.run(buf33, buf35, 16, grid=grid(16), stream=stream0)
        del buf30
        del buf31
        del buf32
        del buf33
        # Topologically Sorted Source Nodes: [z], Original ATen: [aten.complex]
        buf36 = torch.ops.aten.complex.default(buf34, buf35)
        buf37 = buf36
        del buf36
        # Topologically Sorted Source Nodes: [mul], Original ATen: [aten.mul]
        buf38 = torch.ops.aten.mul.Tensor(buf20, primals_2)
        del buf20
        buf39 = buf38
        del buf38
        # Topologically Sorted Source Nodes: [cat_2], Original ATen: [aten.cat]
        buf40 = torch.ops.aten.cat.default([primals_1, buf39], 1)
        del buf39
        del primals_1
        buf41 = buf40
        del buf40
        # Topologically Sorted Source Nodes: [getattr_7], Original ATen: [aten.permute]
        buf42 = torch.ops.aten.permute.default(primals_7, [1, 0])
        buf43 = buf42
        # Topologically Sorted Source Nodes: [matmul_2], Original ATen: [aten.mm]
        buf44 = torch.ops.aten.mm.default(buf41, buf43)
        buf45 = buf44
        del buf44
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
        buf46 = torch.ops.aten.view.dtype(buf45, torch.float32)
        buf47 = buf46
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
        buf48 = torch.ops.aten.view.dtype(primals_8, torch.float32)
        buf49 = buf48
        buf50 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_2.run(buf47, buf49, buf50, 32, grid=grid(32), stream=stream0)
        del buf45
        del buf46
        del buf47
        del buf48
        del buf49
        del primals_8
        # Topologically Sorted Source Nodes: [add_2], Original ATen: [aten.add]
        buf51 = torch.ops.aten.view.dtype(reinterpret_tensor(buf50, (4, 8), (8, 1), 0), torch.complex64)
        buf52 = buf51
        # Topologically Sorted Source Nodes: [getattr_8], Original ATen: [aten.view_as_real]
        buf53 = torch.ops.aten.view_as_real.default(buf52)
        buf54 = buf53
        buf55 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tanh], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_5.run(buf54, buf55, 16, grid=grid(16), stream=stream0)
        buf56 = empty_strided_cuda((4, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tanh_1], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_6.run(buf54, buf56, 16, grid=grid(16), stream=stream0)
        del buf51
        del buf52
        del buf53
        del buf54
        # Topologically Sorted Source Nodes: [h_hat], Original ATen: [aten.complex]
        buf57 = torch.ops.aten.complex.default(buf55, buf56)
        buf58 = buf57
        del buf57
        # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.rsub]
        buf59 = torch.ops.aten.sub.Scalar(1, buf37)
        buf60 = buf59
        del buf59
        # Topologically Sorted Source Nodes: [mul_1], Original ATen: [aten.mul]
        buf61 = torch.ops.aten.mul.Tensor(buf60, buf58)
        buf62 = buf61
        del buf61
        # Topologically Sorted Source Nodes: [mul_2], Original ATen: [aten.mul]
        buf63 = torch.ops.aten.mul.Tensor(buf37, primals_2)
        del buf37
        buf64 = buf63
        del buf63
        # Topologically Sorted Source Nodes: [h_new], Original ATen: [aten.add]
        buf65 = torch.ops.aten.view.dtype(buf62, torch.float32)
        buf66 = buf65
        # Topologically Sorted Source Nodes: [h_new], Original ATen: [aten.add]
        buf67 = torch.ops.aten.view.dtype(buf64, torch.float32)
        buf68 = buf67
        buf69 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [h_new], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_7.run(buf66, buf68, buf69, 32, grid=grid(32), stream=stream0)
        del buf62
        del buf64
        del buf65
        del buf66
        del buf67
        del buf68
        # Topologically Sorted Source Nodes: [h_new], Original ATen: [aten.add]
        buf70 = torch.ops.aten.view.dtype(reinterpret_tensor(buf69, (4, 8), (8, 1), 0), torch.complex64)
        buf71 = buf70
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf72 = torch.ops.aten._conj.default(buf60)
        buf73 = buf72
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf74 = torch.ops.aten._conj.default(buf58)
        buf75 = buf74
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf76 = torch.ops.aten._conj.default(buf41)
        buf77 = buf76
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.t]
        buf78 = torch.ops.aten.permute.default(buf43, [1, 0])
        buf79 = buf78
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf80 = torch.ops.aten._conj.default(buf79)
        buf81 = buf80
        del primals_7
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._conj]
        buf82 = torch.ops.aten._conj.default(buf3)
        buf83 = buf82
    return (buf71, primals_2, buf17, buf18, buf34, buf35, buf55, buf56, buf73, buf75, buf77, buf81, buf83, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.complex64)
    primals_4 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.complex64)
    primals_5 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.complex64)
    primals_6 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.complex64)
    primals_7 = rand_strided((4, 8), (8, 1), device='cuda:0', dtype=torch.complex64)
    primals_8 = rand_strided((1, 4), (4, 1), device='cuda:0', dtype=torch.complex64)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
