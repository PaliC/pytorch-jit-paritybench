# AOT ID: ['167_forward']
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


# kernel path: inductor_cache/c6/cc6rvvlmdy7hes25ayz4s7r4nc7mvjv2njza7v6j6wtmcq52o62p.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_0 = async_compile.triton('triton_poi_fused_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kx/ckxv34dov7yjwqru7abrg3ij3rrfnjedo3pkzmj2gg4vwl6er7bh.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_1 = async_compile.triton('triton_poi_fused_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 25
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 12800*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/2t/c2tkxln7nl4dyjugwg6gcok33ivn6vnnsfkvy4r6v4scycguylia.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_2 = async_compile.triton('triton_poi_fused_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 2097152*y1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/2a/c2a2czyzwgqruq6zhqnxgwrssuifcurd7v5m6krmy7zvr2m57qvw.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_1 => convolution
# Graph fragment:
#   %convolution : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%primals_1, %primals_2, %primals_3, [1, 1], [0, 0], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_3 = async_compile.triton('triton_poi_fused_convolution_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/ke/ckenmkwvanuonpizjvto4eeeb44x7vfr433iijck6unjgnkkmfpt.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 2048, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x2 + 2097152*y1), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1, 1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2 + 4096*y3), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/sf/csfl2fvbsvxkgue3dhnw7k6vvo6mf7nun6zn2habl3f3sr27aa4b.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 25088*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/37/c37qj7dmeimpnleur3hhuqlmk5reqyfbchgdzei3c7gzlojatxyh.py
# Topologically Sorted Source Nodes: [value, value_1, value_2, value_3, mean], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   mean => mean
#   value => add_8
#   value_1 => add_9
#   value_2 => add_10
#   value_3 => add_11
# Graph fragment:
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, 0), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %relu_1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %relu_2), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %relu_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_11, [-1]), kwargs = {})
triton_per_fused_add_mean_6 = async_compile.triton('triton_per_fused_add_mean_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), None)
    tmp3 = tl.load(in_ptr1 + (r1 + 64*x0), None)
    tmp5 = tl.load(in_ptr2 + (r1 + 64*x0), None)
    tmp7 = tl.load(in_ptr3 + (r1 + 64*x0), None)
    tmp1 = 0.0
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp11, None)
''', device_str='cuda')


# kernel path: inductor_cache/up/cupccq4pa3545kzfsi4agdrkgv2drzsgbqutgz2uevjqfzel4m2s.py
# Topologically Sorted Source Nodes: [value, value_1, value_2, value_3, mean, S], Original ATen: [aten.add, aten.mean]
# Source node to ATen node mapping:
#   S => mean_1
#   mean => mean
#   value => add_8
#   value_1 => add_9
#   value_2 => add_10
#   value_3 => add_11
# Graph fragment:
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%relu, 0), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %relu_1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %relu_2), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_10, %relu_3), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%add_11, [-1]), kwargs = {})
#   %mean_1 : [num_users=2] = call_function[target=torch.ops.aten.mean.dim](args = (%mean, [-1]), kwargs = {})
triton_per_fused_add_mean_7 = async_compile.triton('triton_per_fused_add_mean_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_add_mean_7(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + 64*x0), xmask, other=0.0)
    tmp1 = 64.0
    tmp2 = tmp0 / tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = tmp6 / tmp1
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp7, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/km/ckmd2hpsf2v6yr3bwa6mkgatzsnhmtudwq6lcv6swbbvaiy2n5fj.py
# Topologically Sorted Source Nodes: [attention_weughts], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   attention_weughts => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view_1, %view_2, %view_3, %view_4],), kwargs = {})
triton_poi_fused_stack_8 = async_compile.triton('triton_poi_fused_stack_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = xindex // 512
    x0 = (xindex % 512)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 512*(x1)), tmp4, other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 8, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (x0 + 512*((-4) + x1)), tmp9, other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 512*((-8) + x1)), tmp14, other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 16, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (x0 + 512*((-12) + x1)), tmp16, other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, None)
''', device_str='cuda')


# kernel path: inductor_cache/xh/cxhywhdoibdrp3v4yzhbou3y3vhmf6byu5c2ufjrnoh3a25wmkf2.py
# Topologically Sorted Source Nodes: [attention_weughts_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attention_weughts_1 => amax, exp, sub_4
# Graph fragment:
#   %amax : [num_users=1] = call_function[target=torch.ops.aten.amax.default](args = (%view_5, [0], True), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_5, %amax), kwargs = {})
#   %exp : [num_users=2] = call_function[target=torch.ops.aten.exp.default](args = (%sub_4,), kwargs = {})
triton_poi_fused__softmax_9 = async_compile.triton('triton_poi_fused__softmax_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_9(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (4096 + x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (6144 + x0), None, eviction_policy='evict_last')
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp5 = triton_helpers.maximum(tmp3, tmp4)
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = tmp0 - tmp7
    tmp9 = tl_math.exp(tmp8)
    tl.store(out_ptr0 + (x2), tmp9, None)
''', device_str='cuda')


# kernel path: inductor_cache/y5/cy5a5jma7ncj6p3vi7qmkxrxv5mi3523rlcv32k6y3iwzkt5zv5t.py
# Topologically Sorted Source Nodes: [attention_weughts_1], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   attention_weughts_1 => div, sum_1
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [0], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
triton_poi_fused__softmax_10 = async_compile.triton('triton_poi_fused__softmax_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 2048)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (2048 + x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr0 + (4096 + x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (6144 + x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp8 = tmp0 / tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/mx/cmx3gdwi4fbus4ii6clavqlamjnpr26vhqm2ymfssvgyhozd4ucu.py
# Topologically Sorted Source Nodes: [feats], Original ATen: [aten.stack]
# Source node to ATen node mapping:
#   feats => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %relu_1, %relu_2, %relu_3],), kwargs = {})
triton_poi_fused_stack_11 = async_compile.triton('triton_poi_fused_stack_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 8192, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_stack_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 512*x2 + 2097152*y1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/3a/c3api6onjseajvp6qxcdvzkpjeo73u5r6t76ylynvqyrjhahbfcd.py
# Topologically Sorted Source Nodes: [attention_weughts_1, mul, V], Original ATen: [aten._softmax, aten.mul, aten.sum]
# Source node to ATen node mapping:
#   V => sum_2
#   attention_weughts_1 => div, sum_1
#   mul => mul_12
# Graph fragment:
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%exp, [0], True), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp, %sum_1), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %view), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_12, [0]), kwargs = {})
triton_poi_fused__softmax_mul_sum_12 = async_compile.triton('triton_poi_fused__softmax_mul_sum_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16384, 'x': 512}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_mul_sum_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__softmax_mul_sum_12(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y1 = yindex // 4096
    y3 = yindex
    y0 = (yindex % 4096)
    tmp0 = tl.load(in_ptr0 + (x2 + 512*y1), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + 512*y3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2048 + x2 + 512*y1), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (8388608 + x2 + 512*y3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4096 + x2 + 512*y1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (16777216 + x2 + 512*y3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (6144 + x2 + 512*y1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr1 + (25165824 + x2 + 512*y3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 * tmp8
    tmp10 = tmp6 + tmp9
    tmp13 = tmp11 * tmp12
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (y0 + 4096*x2 + 2097152*y1), tmp14, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35 = args
    args.clear()
    assert_size_stride(primals_1, (4, 512, 64, 64), (2097152, 4096, 64, 1))
    assert_size_stride(primals_2, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_3, (512, ), (1, ))
    assert_size_stride(primals_4, (512, ), (1, ))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, ), (1, ))
    assert_size_stride(primals_8, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (512, ), (1, ))
    assert_size_stride(primals_11, (512, ), (1, ))
    assert_size_stride(primals_12, (512, ), (1, ))
    assert_size_stride(primals_13, (512, ), (1, ))
    assert_size_stride(primals_14, (512, 512, 5, 5), (12800, 25, 5, 1))
    assert_size_stride(primals_15, (512, ), (1, ))
    assert_size_stride(primals_16, (512, ), (1, ))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, ), (1, ))
    assert_size_stride(primals_20, (512, 512, 7, 7), (25088, 49, 7, 1))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (512, ), (1, ))
    assert_size_stride(primals_23, (512, ), (1, ))
    assert_size_stride(primals_24, (512, ), (1, ))
    assert_size_stride(primals_25, (512, ), (1, ))
    assert_size_stride(primals_26, (32, 512), (512, 1))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (512, 32), (32, 1))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, 32), (32, 1))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (512, 32), (32, 1))
    assert_size_stride(primals_33, (512, ), (1, ))
    assert_size_stride(primals_34, (512, 32), (32, 1))
    assert_size_stride(primals_35, (512, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((512, 512, 3, 3), (4608, 1, 1536, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_8, buf1, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del primals_8
        buf2 = empty_strided_cuda((512, 512, 5, 5), (12800, 1, 2560, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_14, buf2, 262144, 25, grid=grid(262144, 25), stream=stream0)
        del primals_14
        buf0 = empty_strided_cuda((4, 512, 64, 64), (2097152, 1, 32768, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_1, buf0, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        del primals_1
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf0, primals_2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf5 = buf4; del buf4  # reuse
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf5, primals_3, 8388608, grid=grid(8388608), stream=stream0)
        del primals_3
        buf16 = empty_strided_cuda((16, 512, 64, 64), (2097152, 4096, 64, 1), torch.float32)
        buf6 = reinterpret_tensor(buf16, (4, 512, 64, 64), (2097152, 4096, 64, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf5, primals_4, primals_5, primals_6, primals_7, buf6, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf8 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf8, primals_9, 8388608, grid=grid(8388608), stream=stream0)
        del primals_9
        buf9 = reinterpret_tensor(buf16, (4, 512, 64, 64), (2097152, 4096, 64, 1), 8388608)  # alias
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, primals_10, primals_11, primals_12, primals_13, buf9, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf0, buf2, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf11 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf11, primals_15, 8388608, grid=grid(8388608), stream=stream0)
        del primals_15
        buf12 = reinterpret_tensor(buf16, (4, 512, 64, 64), (2097152, 4096, 64, 1), 16777216)  # alias
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf11, primals_16, primals_17, primals_18, primals_19, buf12, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf3 = empty_strided_cuda((512, 512, 7, 7), (25088, 1, 3584, 512), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_20, buf3, 262144, 49, grid=grid(262144, 49), stream=stream0)
        del primals_20
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf0, buf3, stride=(1, 1), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 512, 64, 64), (2097152, 1, 32768, 512))
        buf14 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [input_10], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_3.run(buf14, primals_21, 8388608, grid=grid(8388608), stream=stream0)
        del primals_21
        buf15 = reinterpret_tensor(buf16, (4, 512, 64, 64), (2097152, 4096, 64, 1), 25165824)  # alias
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf14, primals_22, primals_23, primals_24, primals_25, buf15, 2048, 4096, grid=grid(2048, 4096), stream=stream0)
        buf18 = empty_strided_cuda((4, 512, 64), (32768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [value, value_1, value_2, value_3, mean], Original ATen: [aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_6.run(buf6, buf9, buf12, buf15, buf18, 131072, 64, grid=grid(131072), stream=stream0)
        buf19 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [value, value_1, value_2, value_3, mean, S], Original ATen: [aten.add, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_mean_7.run(buf20, buf18, 2048, 64, grid=grid(2048), stream=stream0)
        del buf18
        buf21 = empty_strided_cuda((4, 32), (32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [Z], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_27, buf20, reinterpret_tensor(primals_26, (512, 32), (1, 512), 0), alpha=1, beta=1, out=buf21)
        del primals_27
        buf22 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_29, buf21, reinterpret_tensor(primals_28, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf22)
        del primals_29
        buf23 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_31, buf21, reinterpret_tensor(primals_30, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf23)
        del primals_31
        buf24 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_2], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_33, buf21, reinterpret_tensor(primals_32, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf24)
        del primals_33
        buf25 = empty_strided_cuda((4, 512), (512, 1), torch.float32)
        # Topologically Sorted Source Nodes: [weight_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_35, buf21, reinterpret_tensor(primals_34, (32, 512), (1, 32), 0), alpha=1, beta=1, out=buf25)
        del primals_35
        buf26 = empty_strided_cuda((16, 512, 1, 1), (512, 1, 512, 512), torch.float32)
        # Topologically Sorted Source Nodes: [attention_weughts], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_8.run(buf22, buf23, buf24, buf25, buf26, 8192, grid=grid(8192), stream=stream0)
        del buf22
        del buf23
        del buf24
        del buf25
        buf27 = empty_strided_cuda((4, 4, 512, 1, 1), (2048, 512, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [attention_weughts_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_9.run(buf26, buf27, 8192, grid=grid(8192), stream=stream0)
        buf28 = empty_strided_cuda((4, 4, 512, 1, 1), (2048, 512, 1, 8192, 8192), torch.float32)
        # Topologically Sorted Source Nodes: [attention_weughts_1], Original ATen: [aten._softmax]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_10.run(buf27, buf28, 8192, grid=grid(8192), stream=stream0)
        del buf27
        buf17 = empty_strided_cuda((16, 512, 64, 64), (2097152, 1, 32768, 512), torch.float32)
        # Topologically Sorted Source Nodes: [feats], Original ATen: [aten.stack]
        stream0 = get_raw_stream(0)
        triton_poi_fused_stack_11.run(buf16, buf17, 8192, 4096, grid=grid(8192, 4096), stream=stream0)
        del buf12
        del buf15
        del buf16
        del buf6
        del buf9
        buf29 = empty_strided_cuda((4, 512, 64, 64), (2097152, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attention_weughts_1, mul, V], Original ATen: [aten._softmax, aten.mul, aten.sum]
        stream0 = get_raw_stream(0)
        triton_poi_fused__softmax_mul_sum_12.run(buf28, buf17, buf29, 16384, 512, grid=grid(16384, 512), stream=stream0)
        del buf28
    return (buf29, buf0, primals_2, primals_4, primals_5, primals_6, primals_7, buf1, primals_10, primals_11, primals_12, primals_13, buf2, primals_16, primals_17, primals_18, primals_19, buf3, primals_22, primals_23, primals_24, primals_25, buf5, buf8, buf11, buf14, buf17, buf20, buf21, buf26, primals_34, primals_32, primals_30, primals_28, primals_26, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 512, 64, 64), (2097152, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((512, 512, 5, 5), (12800, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, 512, 7, 7), (25088, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((512, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
