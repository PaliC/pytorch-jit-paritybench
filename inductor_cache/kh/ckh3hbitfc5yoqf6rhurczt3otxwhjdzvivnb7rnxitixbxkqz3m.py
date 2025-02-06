# AOT ID: ['37_forward']
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


# kernel path: inductor_cache/rh/crhhxsiawubi6ghcklktaq2wuj6jhs2teduw7flej3eaijqtvpwu.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.replication_pad1d]
# Source node to ATen node mapping:
#   input_1 => _unsafe_index
# Graph fragment:
#   %_unsafe_index : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute, [None, None, %clamp_max]), kwargs = {})
triton_poi_fused_replication_pad1d_0 = async_compile.triton('triton_poi_fused_replication_pad1d_0', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_replication_pad1d_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_replication_pad1d_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 4)
    x2 = xindex // 28
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 8*((1) * ((1) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (1))) + 16*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jv/cjvyhu7avkai2j4tldeb5hrnpd564avrthmziuq57bkaxzvvrhmh.py
# Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.replication_pad1d]
# Source node to ATen node mapping:
#   input_7 => _unsafe_index_1
# Graph fragment:
#   %_unsafe_index_1 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%permute_1, [None, None, %clamp_max]), kwargs = {})
triton_poi_fused_replication_pad1d_1 = async_compile.triton('triton_poi_fused_replication_pad1d_1', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_replication_pad1d_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_replication_pad1d_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 4)
    x2 = xindex // 28
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (4 + x1 + 8*((1) * ((1) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (1))) + 16*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/rb/crbnjyjv46kyn233mue6c3kiagnvbjlmecaaa2uvceqeprega3pu.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.convolution, aten.leaky_relu]
# Source node to ATen node mapping:
#   input_2 => convolution
#   input_3 => gt, mul, where
# Graph fragment:
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%_unsafe_index, %primals_2, %primals_3, [1], [0], [1], False, [0], 1), kwargs = {})
#   %gt : [num_users=1] = call_function[target=torch.ops.aten.gt.Scalar](args = (%convolution, 0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convolution, 0.01), kwargs = {})
#   %where : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%gt, %convolution, %mul), kwargs = {})
triton_poi_fused_convolution_leaky_relu_2 = async_compile.triton('triton_poi_fused_convolution_leaky_relu_2', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_leaky_relu_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_leaky_relu_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 4) % 16)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = tmp2 > tmp3
    tmp5 = 0.01
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/e5/ce5ytvmsnxzpmcmraamhgmopf5k2yfuluaka2m6vbcdhyclvq5xn.py
# Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_5 => convolution_1
# Graph fragment:
#   %convolution_1 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where, %primals_4, %primals_5, [1], [0], [1], False, [0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 2) % 4)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f4/cf4v3wkvi6i4nevkjxq2q3omb7ywmjbaqmsthfpfgrhvcsvtlegk.py
# Topologically Sorted Source Nodes: [input_6, exp, d], Original ATen: [aten.tanh, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   d => mul_1
#   exp => exp
#   input_6 => tanh
# Graph fragment:
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_1,), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%tanh,), kwargs = {})
#   %mul_1 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute_1, %exp), kwargs = {})
triton_poi_fused_exp_mul_tanh_4 = async_compile.triton('triton_poi_fused_exp_mul_tanh_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_mul_tanh_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_mul_tanh_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2)
    y1 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (4 + x2 + 8*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 2*x2 + 8*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = libdevice.tanh(tmp1)
    tmp3 = tl_math.exp(tmp2)
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/6g/c6gxosibjg2vhjvjflmaydp3ljenhx4a2xxxg5jdnumfgl6c6kss.py
# Topologically Sorted Source Nodes: [input_12, exp_1, c], Original ATen: [aten.tanh, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   c => mul_3
#   exp_1 => exp_1
#   input_12 => tanh_1
# Graph fragment:
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_3,), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%tanh_1,), kwargs = {})
#   %mul_3 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%permute, %exp_1), kwargs = {})
triton_poi_fused_exp_mul_tanh_5 = async_compile.triton('triton_poi_fused_exp_mul_tanh_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 4}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_exp_mul_tanh_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_exp_mul_tanh_5(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 2)
    y1 = yindex // 2
    tmp0 = tl.load(in_ptr0 + (x2 + 8*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + 2*x2 + 8*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = libdevice.tanh(tmp1)
    tmp3 = tl_math.exp(tmp2)
    tmp4 = tmp0 * tmp3
    tl.store(out_ptr0 + (x2 + 4*y3), tmp4, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ak/cak3bccmezejvwlolhfe5bymexymkgct4sduqfy3k3apmy2oxarp.py
# Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.replication_pad1d]
# Source node to ATen node mapping:
#   input_13 => _unsafe_index_2
# Graph fragment:
#   %_unsafe_index_2 : [num_users=2] = call_function[target=torch.ops.aten._unsafe_index.Tensor](args = (%mul_1, [None, None, %clamp_max]), kwargs = {})
triton_poi_fused_replication_pad1d_6 = async_compile.triton('triton_poi_fused_replication_pad1d_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_replication_pad1d_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_replication_pad1d_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 4)
    x2 = xindex // 28
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 4*((1) * ((1) <= (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0))))) + (((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) * ((((0) * ((0) >= ((-2) + x0)) + ((-2) + x0) * (((-2) + x0) > (0)))) < (1))) + 8*x2), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x3), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ov/covnecl6cfzd3cttbdvjnck6sus3mrq7wm24rbgskxkt73l3bc7r.py
# Topologically Sorted Source Nodes: [input_17, input_18, x_even_update], Original ATen: [aten.convolution, aten.tanh, aten.add]
# Source node to ATen node mapping:
#   input_17 => convolution_5
#   input_18 => tanh_2
#   x_even_update => add
# Graph fragment:
#   %convolution_5 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_2, %primals_12, %primals_13, [1], [0], [1], False, [0], 1), kwargs = {})
#   %tanh_2 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_5,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %tanh_2), kwargs = {})
triton_poi_fused_add_convolution_tanh_7 = async_compile.triton('triton_poi_fused_add_convolution_tanh_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 2}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_tanh_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_tanh_7(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 2
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2 + 2*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 4*x2 + 8*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = libdevice.tanh(tmp2)
    tmp5 = tmp3 + tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 2*y3), tmp2, xmask & ymask)
    tl.store(out_ptr0 + (x2 + 2*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/hn/chndpzkqgyr3pnyflxn2yx42xn4agczldnhcqf53vrp7foja3kss.py
# Topologically Sorted Source Nodes: [input_23, input_24, x_odd_update], Original ATen: [aten.convolution, aten.tanh, aten.sub]
# Source node to ATen node mapping:
#   input_23 => convolution_7
#   input_24 => tanh_3
#   x_odd_update => sub
# Graph fragment:
#   %convolution_7 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%where_3, %primals_16, %primals_17, [1], [0], [1], False, [0], 1), kwargs = {})
#   %tanh_3 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_7,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_1, %tanh_3), kwargs = {})
triton_poi_fused_convolution_sub_tanh_8 = async_compile.triton('triton_poi_fused_convolution_sub_tanh_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 2}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_sub_tanh_8', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_sub_tanh_8(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 2
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    tmp0 = tl.load(in_out_ptr0 + (x2 + 2*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0 + 4*x2 + 8*y1), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = libdevice.tanh(tmp2)
    tmp5 = tmp3 - tmp4
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 2*y3), tmp2, xmask & ymask)
    tl.store(out_ptr0 + (x2 + 2*y3), tmp5, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxqy6jdalupboh6hdksh2yex2mbssuag2m4jgr2fimyohectgjn.py
# Topologically Sorted Source Nodes: [input_18, x_even_update, permute_2], Original ATen: [aten.tanh, aten.add, aten.permute]
# Source node to ATen node mapping:
#   input_18 => tanh_2
#   permute_2 => permute_2
#   x_even_update => add
# Graph fragment:
#   %tanh_2 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%convolution_5,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_3, %tanh_2), kwargs = {})
#   %permute_2 : [num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%add, [0, 2, 1]), kwargs = {})
triton_poi_fused_add_permute_tanh_9 = async_compile.triton('triton_poi_fused_add_permute_tanh_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8, 'x': 4}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_permute_tanh_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_permute_tanh_9(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8
    xnumel = 4
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 2)
    y1 = yindex // 2
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 2*x2 + 8*y1), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2 + 4*y3), tmp0, xmask & ymask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4), (16, 4, 1))
    assert_size_stride(primals_2, (16, 4, 4), (16, 4, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (4, 16, 3), (48, 3, 1))
    assert_size_stride(primals_5, (4, ), (1, ))
    assert_size_stride(primals_6, (16, 4, 4), (16, 4, 1))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (4, 16, 3), (48, 3, 1))
    assert_size_stride(primals_9, (4, ), (1, ))
    assert_size_stride(primals_10, (16, 4, 4), (16, 4, 1))
    assert_size_stride(primals_11, (16, ), (1, ))
    assert_size_stride(primals_12, (4, 16, 3), (48, 3, 1))
    assert_size_stride(primals_13, (4, ), (1, ))
    assert_size_stride(primals_14, (16, 4, 4), (16, 4, 1))
    assert_size_stride(primals_15, (16, ), (1, ))
    assert_size_stride(primals_16, (4, 16, 3), (48, 3, 1))
    assert_size_stride(primals_17, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 7), (28, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.replication_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad1d_0.run(primals_1, buf0, 112, grid=grid(112), stream=stream0)
        buf6 = empty_strided_cuda((4, 4, 7), (28, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.replication_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad1d_1.run(primals_1, buf6, 112, grid=grid(112), stream=stream0)
        # Topologically Sorted Source Nodes: [input_2], Original ATen: [aten.convolution]
        buf1 = extern_kernels.convolution(buf0, primals_2, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf1, (4, 16, 4), (64, 4, 1))
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_6, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf7, (4, 16, 4), (64, 4, 1))
        buf2 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_2.run(buf2, primals_3, 256, grid=grid(256), stream=stream0)
        del primals_3
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_2.run(buf8, primals_7, 256, grid=grid(256), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, primals_4, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf3, (4, 4, 2), (8, 2, 1))
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_8, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf9, (4, 4, 2), (8, 2, 1))
        buf4 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [input_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf4, primals_5, 32, grid=grid(32), stream=stream0)
        del primals_5
        buf10 = buf9; del buf9  # reuse
        # Topologically Sorted Source Nodes: [input_11], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf10, primals_9, 32, grid=grid(32), stream=stream0)
        del primals_9
        buf5 = empty_strided_cuda((4, 4, 2), (8, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, exp, d], Original ATen: [aten.tanh, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_tanh_4.run(primals_1, buf4, buf5, 8, 4, grid=grid(8, 4), stream=stream0)
        buf11 = empty_strided_cuda((4, 4, 2), (8, 1, 4), torch.float32)
        # Topologically Sorted Source Nodes: [input_12, exp_1, c], Original ATen: [aten.tanh, aten.exp, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_exp_mul_tanh_5.run(primals_1, buf10, buf11, 8, 4, grid=grid(8, 4), stream=stream0)
        buf12 = empty_strided_cuda((4, 4, 7), (28, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten.replication_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad1d_6.run(buf5, buf12, 112, grid=grid(112), stream=stream0)
        buf17 = empty_strided_cuda((4, 4, 7), (28, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_19], Original ATen: [aten.replication_pad1d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_replication_pad1d_6.run(buf11, buf17, 112, grid=grid(112), stream=stream0)
        # Topologically Sorted Source Nodes: [input_14], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_10, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf13, (4, 16, 4), (64, 4, 1))
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_14, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf18, (4, 16, 4), (64, 4, 1))
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_2.run(buf14, primals_11, 256, grid=grid(256), stream=stream0)
        del primals_11
        buf19 = buf18; del buf18  # reuse
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten.convolution, aten.leaky_relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_leaky_relu_2.run(buf19, primals_15, 256, grid=grid(256), stream=stream0)
        del primals_15
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_12, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf15, (4, 4, 2), (8, 2, 1))
        buf16 = buf15; del buf15  # reuse
        buf22 = empty_strided_cuda((4, 4, 2), (8, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_17, input_18, x_even_update], Original ATen: [aten.convolution, aten.tanh, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_tanh_7.run(buf16, primals_13, buf11, buf22, 16, 2, grid=grid(16, 2), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_16, stride=(1,), padding=(0,), dilation=(1,), transposed=False, output_padding=(0,), groups=1, bias=None)
        assert_size_stride(buf20, (4, 4, 2), (8, 2, 1))
        buf21 = buf20; del buf20  # reuse
        buf24 = empty_strided_cuda((4, 4, 2), (8, 2, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_23, input_24, x_odd_update], Original ATen: [aten.convolution, aten.tanh, aten.sub]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_sub_tanh_8.run(buf21, primals_17, buf5, buf24, 16, 2, grid=grid(16, 2), stream=stream0)
        del primals_17
        buf23 = empty_strided_cuda((4, 2, 4), (8, 4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, x_even_update, permute_2], Original ATen: [aten.tanh, aten.add, aten.permute]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_permute_tanh_9.run(buf22, buf23, 8, 4, grid=grid(8, 4), stream=stream0)
        buf25 = reinterpret_tensor(buf22, (4, 2, 4), (8, 4, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [input_24, x_odd_update, permute_3], Original ATen: [aten.tanh, aten.sub, aten.permute]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_permute_tanh_9.run(buf24, buf25, 8, 4, grid=grid(8, 4), stream=stream0)
        del buf24
    return (buf23, buf25, primals_2, primals_4, primals_6, primals_8, primals_10, primals_12, primals_14, primals_16, reinterpret_tensor(primals_1, (4, 4, 2), (16, 1, 8), 0), reinterpret_tensor(primals_1, (4, 4, 2), (16, 1, 8), 4), buf0, buf2, buf4, buf5, buf6, buf8, buf10, buf11, buf12, buf14, buf16, buf17, buf19, buf21, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 16, 3), (48, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((4, 16, 3), (48, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4, 16, 3), (48, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((16, 4, 4), (16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((4, 16, 3), (48, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
