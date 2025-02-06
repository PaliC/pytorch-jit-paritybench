# AOT ID: ['23_inference']
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


# kernel path: inductor_cache/2m/c2meqvmgkfgmyl6mqcuwhzlohkljnsusvq6645f33xlovyoeioy5.py
# Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   einsum => bmm
# Graph fragment:
#   %bmm : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_2, %view_3), kwargs = {})
triton_poi_fused_bmm_0 = async_compile.triton('triton_poi_fused_bmm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qr/cqr22v234l4i7fj2f63q6muflpcbyumxvqfeahafjj6aenrn3qmp.py
# Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   einsum_1 => bmm_1
# Graph fragment:
#   %bmm_1 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_8, %view_9), kwargs = {})
triton_poi_fused_bmm_1 = async_compile.triton('triton_poi_fused_bmm_1', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jn/cjn5v4qgbzpmmph4o4khxsizqvehiqkijprhfxkdnn4ensb74ifl.py
# Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   sub => sub
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %view_11), kwargs = {})
triton_poi_fused_sub_2 = async_compile.triton('triton_poi_fused_sub_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sub_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5r/c5r2nz7s3jeabik3rkdnz6cmajm2c7ohj7ppmfwwxfehvn7fsbwv.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_17, %view_23), kwargs = {})
triton_poi_fused_add_3 = async_compile.triton('triton_poi_fused_add_3', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/th/cthbhpcnmre4uqces5dvhjf4342xgqxx4qqwcg6khg2znaqgbdfm.py
# Topologically Sorted Source Nodes: [tanh], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   tanh => tanh
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%select_30,), kwargs = {})
triton_poi_fused_tanh_4 = async_compile.triton('triton_poi_fused_tanh_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/5q/c5q6ddnf6p5wqlxarxj5kjymzpr5fqegdn7janyhxz5r74xnhuzh.py
# Topologically Sorted Source Nodes: [tanh_1], Original ATen: [aten.tanh]
# Source node to ATen node mapping:
#   tanh_1 => tanh_1
# Graph fragment:
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%select_31,), kwargs = {})
triton_poi_fused_tanh_5 = async_compile.triton('triton_poi_fused_tanh_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_tanh_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_tanh_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tmp1 = libdevice.tanh(tmp0)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/6t/c6trtu77vdygzc4hqvlzqcldpwemap2axmlteb3ipyg4pc4uneam.py
# Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   einsum_4 => bmm_4
# Graph fragment:
#   %bmm_4 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_24, %view_26), kwargs = {})
triton_poi_fused_bmm_6 = async_compile.triton('triton_poi_fused_bmm_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qp/cqpnrorczlj4ubk6abkpohmyhpwp7pxby7c2v3mfmpodgwez7aan.py
# Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.bmm]
# Source node to ATen node mapping:
#   einsum_5 => bmm_5
# Graph fragment:
#   %bmm_5 : [num_users=1] = call_function[target=torch.ops.aten.bmm.default](args = (%view_29, %view_31), kwargs = {})
triton_poi_fused_bmm_7 = async_compile.triton('triton_poi_fused_bmm_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_bmm_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_bmm_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (1 + 2*x0), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/sc/cschtdbe2fctkufxisscrttzjttytw32mr7t3pv2kos5ftomp5wg.py
# Topologically Sorted Source Nodes: [sub_1], Original ATen: [aten.sub]
# Source node to ATen node mapping:
#   sub_1 => sub_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_28, %view_33), kwargs = {})
triton_poi_fused_sub_8 = async_compile.triton('triton_poi_fused_sub_8', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sub_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_sub_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 - tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rp/crpy5urhr2xj7jt4zhpus2swrd7r7cdrtpkx2lbguwiampsxtwio.py
# Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_1 => add_1
# Graph fragment:
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_38, %view_43), kwargs = {})
triton_poi_fused_add_9 = async_compile.triton('triton_poi_fused_add_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_9(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg1_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(arg2_1, (4, 4, 4, 4), (64, 16, 4, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # Topologically Sorted Source Nodes: [xq_ft_], Original ATen: [aten.zeros]
        buf4 = torch.ops.aten.full.default([4, 4, 4, 2], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf5 = buf4
        del buf4
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.select]
        buf6 = torch.ops.aten.select.int(buf5, 3, 0)
        buf7 = buf6
        # Topologically Sorted Source Nodes: [xk_ft_], Original ATen: [aten.zeros]
        buf26 = torch.ops.aten.full.default([4, 4, 4, 2], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf27 = buf26
        del buf26
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.select]
        buf28 = torch.ops.aten.select.int(buf27, 3, 0)
        buf29 = buf28
        # Topologically Sorted Source Nodes: [out_ft], Original ATen: [aten.zeros]
        buf0 = torch.ops.aten.full.default([4, 4, 4, 3], 0, dtype=torch.complex64, layout=torch.strided, device=device(type='cuda', index=0), pin_memory=False)
        buf1 = buf0
        del buf0
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.select]
        buf2 = torch.ops.aten.select.int(buf1, 3, 0)
        buf3 = buf2
        # Topologically Sorted Source Nodes: [xq_ft], Original ATen: [aten._fft_r2c]
        buf8 = torch.ops.aten._fft_r2c.default(reinterpret_tensor(arg0_1, (4, 4, 4, 4), (64, 1, 4, 16), 0), [3], 0, True)
        del arg0_1
        buf9 = buf8
        del buf8
        # Topologically Sorted Source Nodes: [getitem], Original ATen: [aten.select]
        buf10 = torch.ops.aten.select.int(buf9, 3, 0)
        buf11 = buf10
        # Topologically Sorted Source Nodes: [setitem], Original ATen: [aten.copy]
        buf12 = torch.ops.aten.copy.default(buf7, buf11)
        del buf10
        del buf11
        del buf6
        del buf7
        # Topologically Sorted Source Nodes: [getitem_1], Original ATen: [aten.select]
        buf18 = torch.ops.aten.select.int(buf9, 3, 1)
        buf13 = buf12
        del buf12
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf14 = torch.ops.aten.select_scatter.default(buf5, buf13, 3, 0)
        del buf13
        del buf5
        buf15 = buf14
        del buf14
        buf19 = buf18
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.select]
        buf16 = torch.ops.aten.select.int(buf15, 3, 1)
        buf17 = buf16
        # Topologically Sorted Source Nodes: [setitem_1], Original ATen: [aten.copy]
        buf20 = torch.ops.aten.copy.default(buf17, buf19)
        del buf16
        del buf17
        del buf18
        del buf19
        del buf9
        buf21 = buf20
        del buf20
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf22 = torch.ops.aten.select_scatter.default(buf15, buf21, 3, 1)
        del buf15
        del buf21
        buf23 = buf22
        del buf22
        # Topologically Sorted Source Nodes: [xk_ft], Original ATen: [aten._fft_r2c]
        buf30 = torch.ops.aten._fft_r2c.default(reinterpret_tensor(arg1_1, (4, 4, 4, 4), (64, 1, 4, 16), 0), [3], 0, True)
        del arg1_1
        buf31 = buf30
        del buf30
        # Topologically Sorted Source Nodes: [getitem_2], Original ATen: [aten.select]
        buf32 = torch.ops.aten.select.int(buf31, 3, 0)
        buf33 = buf32
        # Topologically Sorted Source Nodes: [setitem_2], Original ATen: [aten.copy]
        buf34 = torch.ops.aten.copy.default(buf29, buf33)
        del buf28
        del buf29
        del buf32
        del buf33
        # Topologically Sorted Source Nodes: [getitem_3], Original ATen: [aten.select]
        buf40 = torch.ops.aten.select.int(buf31, 3, 1)
        buf35 = buf34
        del buf34
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf36 = torch.ops.aten.select_scatter.default(buf27, buf35, 3, 0)
        del buf27
        del buf35
        buf37 = buf36
        del buf36
        buf41 = buf40
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.select]
        buf38 = torch.ops.aten.select.int(buf37, 3, 1)
        buf39 = buf38
        # Topologically Sorted Source Nodes: [setitem_3], Original ATen: [aten.copy]
        buf42 = torch.ops.aten.copy.default(buf39, buf41)
        del buf31
        del buf38
        del buf39
        del buf40
        del buf41
        buf43 = buf42
        del buf42
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf44 = torch.ops.aten.select_scatter.default(buf37, buf43, 3, 1)
        del buf37
        del buf43
        buf45 = buf44
        del buf44
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.view_as_real]
        buf24 = torch.ops.aten.view_as_real.default(buf23)
        buf25 = buf24
        buf48 = empty_strided_cuda((16, 2, 4), (8, 1, 2), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_0.run(buf25, buf48, 128, grid=grid(128), stream=stream0)
        del buf24
        del buf25
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.view_as_real]
        buf51 = torch.ops.aten.view_as_real.default(buf23)
        buf52 = buf51
        buf55 = empty_strided_cuda((16, 2, 4), (8, 1, 2), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_1.run(buf52, buf55, 128, grid=grid(128), stream=stream0)
        del buf51
        del buf52
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.view_as_real]
        buf58 = torch.ops.aten.view_as_real.default(buf23)
        buf59 = buf58
        buf62 = empty_strided_cuda((16, 2, 4), (8, 1, 2), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_0.run(buf59, buf62, 128, grid=grid(128), stream=stream0)
        del buf58
        del buf59
        # Topologically Sorted Source Nodes: [einsum_3], Original ATen: [aten.view_as_real]
        buf65 = torch.ops.aten.view_as_real.default(buf23)
        buf66 = buf65
        buf69 = empty_strided_cuda((16, 2, 4), (8, 1, 2), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_3], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_1.run(buf66, buf69, 128, grid=grid(128), stream=stream0)
        del buf23
        del buf65
        del buf66
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.view_as_real]
        buf46 = torch.ops.aten.view_as_real.default(buf45)
        buf47 = buf46
        buf49 = empty_strided_cuda((16, 4, 2), (8, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_0.run(buf47, buf49, 128, grid=grid(128), stream=stream0)
        del buf46
        del buf47
        buf50 = empty_strided_cuda((16, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf48, buf49, out=buf50)
        del buf48
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.view_as_real]
        buf53 = torch.ops.aten.view_as_real.default(buf45)
        buf54 = buf53
        buf56 = buf49; del buf49  # reuse
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_1.run(buf54, buf56, 128, grid=grid(128), stream=stream0)
        del buf53
        del buf54
        buf57 = empty_strided_cuda((16, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf55, buf56, out=buf57)
        buf72 = reinterpret_tensor(buf50, (4, 4, 2, 2), (16, 4, 2, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [sub], Original ATen: [aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sub_2.run(buf72, buf57, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.view_as_real]
        buf60 = torch.ops.aten.view_as_real.default(buf45)
        buf61 = buf60
        buf63 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_1.run(buf61, buf63, 128, grid=grid(128), stream=stream0)
        del buf60
        del buf61
        buf64 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [einsum_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf62, buf63, out=buf64)
        # Topologically Sorted Source Nodes: [einsum_3], Original ATen: [aten.view_as_real]
        buf67 = torch.ops.aten.view_as_real.default(buf45)
        buf68 = buf67
        buf70 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [einsum_3], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_0.run(buf68, buf70, 128, grid=grid(128), stream=stream0)
        del buf67
        del buf68
        buf71 = empty_strided_cuda((16, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf69, buf70, out=buf71)
        buf73 = reinterpret_tensor(buf64, (4, 4, 2, 2), (16, 4, 2, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_3.run(buf73, buf71, 64, grid=grid(64), stream=stream0)
        # Topologically Sorted Source Nodes: [sub, add, xqk_ft], Original ATen: [aten.sub, aten.add, aten.complex]
        buf74 = torch.ops.aten.complex.default(buf72, buf73)
        buf75 = buf74
        del buf74
        # Topologically Sorted Source Nodes: [getattr_9], Original ATen: [aten.view_as_real]
        buf76 = torch.ops.aten.view_as_real.default(buf75)
        buf77 = buf76
        buf80 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [tanh], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_4.run(buf77, buf80, 64, grid=grid(64), stream=stream0)
        del buf76
        del buf77
        # Topologically Sorted Source Nodes: [getattr_10], Original ATen: [aten.view_as_real]
        buf78 = torch.ops.aten.view_as_real.default(buf75)
        buf79 = buf78
        buf81 = buf72; del buf72  # reuse
        # Topologically Sorted Source Nodes: [tanh_1], Original ATen: [aten.tanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused_tanh_5.run(buf79, buf81, 64, grid=grid(64), stream=stream0)
        del buf75
        del buf78
        del buf79
        # Topologically Sorted Source Nodes: [tanh, tanh_1, xqk_ft_1], Original ATen: [aten.tanh, aten.complex]
        buf82 = torch.ops.aten.complex.default(buf80, buf81)
        buf83 = buf82
        del buf82
        # Topologically Sorted Source Nodes: [getattr_11], Original ATen: [aten.view_as_real]
        buf84 = torch.ops.aten.view_as_real.default(buf83)
        buf85 = buf84
        buf88 = reinterpret_tensor(buf81, (16, 2, 2), (4, 2, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_6.run(buf85, buf88, 64, grid=grid(64), stream=stream0)
        del buf84
        del buf85
        # Topologically Sorted Source Nodes: [getattr_13], Original ATen: [aten.view_as_real]
        buf91 = torch.ops.aten.view_as_real.default(buf83)
        buf92 = buf91
        buf95 = reinterpret_tensor(buf80, (16, 2, 2), (4, 2, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_7.run(buf92, buf95, 64, grid=grid(64), stream=stream0)
        del buf91
        del buf92
        # Topologically Sorted Source Nodes: [getattr_15], Original ATen: [aten.view_as_real]
        buf98 = torch.ops.aten.view_as_real.default(buf83)
        buf99 = buf98
        buf102 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_6.run(buf99, buf102, 64, grid=grid(64), stream=stream0)
        del buf98
        del buf99
        # Topologically Sorted Source Nodes: [getattr_17], Original ATen: [aten.view_as_real]
        buf105 = torch.ops.aten.view_as_real.default(buf83)
        buf106 = buf105
        buf109 = empty_strided_cuda((16, 2, 2), (4, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [einsum_7], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_7.run(buf106, buf109, 64, grid=grid(64), stream=stream0)
        del buf105
        del buf106
        del buf83
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.view_as_real]
        buf86 = torch.ops.aten.view_as_real.default(buf45)
        buf87 = buf86
        buf89 = reinterpret_tensor(buf70, (16, 2, 4), (8, 1, 2), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_0.run(buf87, buf89, 128, grid=grid(128), stream=stream0)
        del buf86
        del buf87
        buf90 = reinterpret_tensor(buf69, (16, 2, 4), (8, 4, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [einsum_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf88, buf89, out=buf90)
        del buf88
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.view_as_real]
        buf93 = torch.ops.aten.view_as_real.default(buf45)
        buf94 = buf93
        buf96 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_1.run(buf94, buf96, 128, grid=grid(128), stream=stream0)
        del buf93
        del buf94
        buf97 = reinterpret_tensor(buf62, (16, 2, 4), (8, 4, 1), 0); del buf62  # reuse
        # Topologically Sorted Source Nodes: [einsum_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf95, buf96, out=buf97)
        del buf95
        buf112 = reinterpret_tensor(buf90, (4, 4, 4, 2), (32, 8, 1, 4), 0); del buf90  # reuse
        # Topologically Sorted Source Nodes: [sub_1], Original ATen: [aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_sub_8.run(buf112, buf97, 128, grid=grid(128), stream=stream0)
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.view_as_real]
        buf100 = torch.ops.aten.view_as_real.default(buf45)
        buf101 = buf100
        buf103 = reinterpret_tensor(buf97, (16, 2, 4), (8, 1, 2), 0); del buf97  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_1.run(buf101, buf103, 128, grid=grid(128), stream=stream0)
        del buf100
        del buf101
        buf104 = reinterpret_tensor(buf96, (16, 2, 4), (8, 4, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [einsum_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf102, buf103, out=buf104)
        del buf102
        # Topologically Sorted Source Nodes: [einsum_7], Original ATen: [aten.view_as_real]
        buf107 = torch.ops.aten.view_as_real.default(buf45)
        buf108 = buf107
        buf110 = buf103; del buf103  # reuse
        # Topologically Sorted Source Nodes: [einsum_7], Original ATen: [aten.bmm]
        stream0 = get_raw_stream(0)
        triton_poi_fused_bmm_0.run(buf108, buf110, 128, grid=grid(128), stream=stream0)
        del buf107
        del buf108
        del buf45
        buf111 = reinterpret_tensor(buf55, (16, 2, 4), (8, 4, 1), 0); del buf55  # reuse
        # Topologically Sorted Source Nodes: [einsum_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(buf109, buf110, out=buf111)
        del buf109
        del buf110
        buf113 = reinterpret_tensor(buf104, (4, 4, 4, 2), (32, 8, 1, 4), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [add_1], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_9.run(buf113, buf111, 128, grid=grid(128), stream=stream0)
        del buf111
        # Topologically Sorted Source Nodes: [sub_1, add_1, xqkv_ft], Original ATen: [aten.sub, aten.add, aten.complex]
        buf114 = torch.ops.aten.complex.default(buf112, buf113)
        del buf112
        del buf113
        buf115 = buf114
        del buf114
        # Topologically Sorted Source Nodes: [getitem_4], Original ATen: [aten.select]
        buf116 = torch.ops.aten.select.int(buf115, 3, 0)
        buf117 = buf116
        # Topologically Sorted Source Nodes: [setitem_4], Original ATen: [aten.copy]
        buf118 = torch.ops.aten.copy.default(buf3, buf117)
        del buf116
        del buf117
        del buf2
        del buf3
        buf119 = buf118
        del buf118
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf120 = torch.ops.aten.select_scatter.default(buf1, buf119, 3, 0)
        del buf1
        del buf119
        buf121 = buf120
        del buf120
        # Topologically Sorted Source Nodes: [getitem_5], Original ATen: [aten.select]
        buf124 = torch.ops.aten.select.int(buf115, 3, 1)
        buf125 = buf124
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.select]
        buf122 = torch.ops.aten.select.int(buf121, 3, 1)
        buf123 = buf122
        # Topologically Sorted Source Nodes: [setitem_5], Original ATen: [aten.copy]
        buf126 = torch.ops.aten.copy.default(buf123, buf125)
        del buf115
        del buf122
        del buf123
        del buf124
        del buf125
        buf127 = buf126
        del buf126
        # Topologically Sorted Source Nodes: [], Original ATen: []
        buf128 = torch.ops.aten.select_scatter.default(buf121, buf127, 3, 1)
        del buf121
        del buf127
        buf129 = buf128
        del buf128
        # Topologically Sorted Source Nodes: [truediv], Original ATen: [aten.div]
        buf130 = torch.ops.aten.div.Scalar(buf129, 4)
        del buf129
        buf131 = buf130
        del buf130
        # Topologically Sorted Source Nodes: [truediv_1], Original ATen: [aten.div]
        buf132 = torch.ops.aten.div.Scalar(buf131, 4)
        del buf131
        buf133 = buf132
        del buf132
        # Topologically Sorted Source Nodes: [fft_irfft], Original ATen: [aten._fft_c2r]
        buf134 = torch.ops.aten._fft_c2r.default(buf133, [3], 2, 4)
        del buf133
        buf135 = buf134
        del buf134
    return (reinterpret_tensor(buf135, (4, 4, 4, 4), (64, 1, 4, 16), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1, arg1_1, arg2_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
