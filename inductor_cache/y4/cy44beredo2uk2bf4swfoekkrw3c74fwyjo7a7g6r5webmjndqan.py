# AOT ID: ['4_forward']
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


# kernel path: inductor_cache/3h/c3hpecenggnrxcqg3nyshgwbawgb3vfvl2axvlomnkql4xptnmer.py
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
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 27*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/nu/cnuedntizcvgrf3a7y7wp5fpvjeo2fjv3uant6ubkp6mi5baj43r.py
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
    size_hints={'y': 4096, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 16)
    y1 = yindex // 16
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 16*x2 + 144*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/oz/cozh5trwm6hhwvruivlyweoxfyar652qvc5cihmjxdqtz42afxn5.py
# Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
# Source node to ATen node mapping:
#   sub => sub
#   x => div
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%primals_2, %primals_1), kwargs = {})
#   %div : [num_users=2] = call_function[target=torch.ops.aten.div.Tensor](args = (%sub, %primals_3), kwargs = {})
triton_poi_fused_div_sub_2 = async_compile.triton('triton_poi_fused_div_sub_2', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 5), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sub_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_div_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = tmp2 / tmp3
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp4, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/g7/cg7p6aeufjzngpfiqwhqagmfmq6vf7rwdyfgjvs2p3sid6rfwxjy.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 160)
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jy/cjyzv4kid2kixanfmyatpjtkuptthvszlllh5ts2elv4tq5xkmoy.py
# Topologically Sorted Source Nodes: [batch_norm, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm => add_1, mul_1, mul_2, sub_1
#   x_1 => relu
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_4', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xu/cxuwwwevqfpeajnxeihxgtgm55vjbpddhmfweyxcnkoe2fydvb7u.py
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
    size_hints={'y': 65536, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 51200
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 160)
    y1 = yindex // 160
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/aj/cajqe2atpl27hplndqpuigcxgwjzqxwf7io3n4xmyrla7xihx57m.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 131072, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/om/comrmhq7asu45rs235ws36rmy2ungqgese7ct3hphrabzgj4lj67.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_7 = async_compile.triton('triton_poi_fused_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 204800
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 320)
    y1 = yindex // 320
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/vv/cvvt7bifexqy7hmmefzaxz6mz44botiqymmphvsgxyel4an64fpb.py
# Topologically Sorted Source Nodes: [batch_norm_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_1 => add_3, mul_4, mul_5, sub_2
#   out_1 => relu_1
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/jt/cjtufbr73oejihaj5euxz5z32slvfcv5bast5khguj4lioxqtfyu.py
# Topologically Sorted Source Nodes: [input_1, batch_norm_2, out_3], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_2 => add_6, mul_7, mul_8, sub_3
#   input_1 => add_4
#   out_3 => relu_2
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_6,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
''', device_str='cuda')


# kernel path: inductor_cache/gr/cgr5wxhpyzymo7x45robds6uskeuprjnkvw2fgfs7ylmn6ylkdr7.py
# Topologically Sorted Source Nodes: [input_1, input_2, batch_norm_4, out_6], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_4 => add_11, mul_13, mul_14, sub_5
#   input_1 => add_4
#   input_2 => add_9
#   out_6 => relu_4
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_5), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_11,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x2), tmp21, None)
''', device_str='cuda')


# kernel path: inductor_cache/63/c63mlpy5mxl7dffas4xmb6zutbhir4gx36et6grrm744ojmgrsky.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, batch_norm_6, out_9], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_6 => add_16, mul_19, mul_20, sub_7
#   input_1 => add_4
#   input_2 => add_9
#   input_3 => add_14
#   out_9 => relu_6
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_5), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_7), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_14, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_16,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x2), tmp23, None)
''', device_str='cuda')


# kernel path: inductor_cache/zw/czw36r65ync2ijm5xknvoq5tvx6vayz7qaa6m326atnov3375mrb.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_12 = async_compile.triton('triton_poi_fused_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 524288, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 409600
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 640)
    y1 = yindex // 640
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 640*x2 + 5760*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/tv/ctvt5uvkymicf5pzeh5fn3ppriviymgasqdbhag6amzp73dznmf7.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, batch_norm_8, out_12], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_8 => add_21, mul_25, mul_26, sub_9
#   input_1 => add_4
#   input_2 => add_9
#   input_3 => add_14
#   input_4 => add_19
#   out_12 => relu_8
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_5), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_7), kwargs = {})
#   %add_19 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_14, %convolution_9), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_21,), kwargs = {})
#   %sub_54 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_19, %unsqueeze_514), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None)
    tmp3 = tl.load(in_ptr2 + (x0), None)
    tmp5 = tl.load(in_ptr3 + (x0), None)
    tmp7 = tl.load(in_out_ptr0 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp25, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/f6/cf6xmwn5eifvu7t635zntkda353cap3jnphdt5lbhehd5ckvikrn.py
# Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.add, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   input_1 => add_4
#   input_2 => add_9
#   input_3 => add_14
# Graph fragment:
#   %add_4 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_3, %convolution_2), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_4, %convolution_5), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %convolution_7), kwargs = {})
#   %sub_56 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_14, %unsqueeze_538), kwargs = {})
#   %sub_58 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %unsqueeze_562), kwargs = {})
#   %sub_60 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_4, %unsqueeze_586), kwargs = {})
triton_poi_fused_add_native_batch_norm_backward_14 = async_compile.triton('triton_poi_fused_add_native_batch_norm_backward_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_batch_norm_backward_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_native_batch_norm_backward_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_out_ptr0 + (x2), None)
    tmp7 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = tmp4 - tmp9
    tmp12 = tmp2 - tmp11
    tl.store(in_out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr0 + (x2), tmp10, None)
    tl.store(out_ptr1 + (x2), tmp12, None)
''', device_str='cuda')


# kernel path: inductor_cache/ry/cryc4y5vnqfwyy5zrme6hk2krhxfuik4sgdocs5vwklahn3v2gmn.py
# Topologically Sorted Source Nodes: [input_5, batch_norm_10, x_2], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_10 => add_26, mul_31, mul_32, sub_11
#   input_5 => add_24
#   x_2 => relu_10
# Graph fragment:
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %convolution_11), kwargs = {})
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_10 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
#   %sub_52 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_24, %unsqueeze_490), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2621440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 160)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/4j/c4j7lvwggmp2clwxy4ib32lyo2wjsnmm5q7nsdndhhdjgri3jexn.py
# Topologically Sorted Source Nodes: [batch_norm_11, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_11 => add_28, mul_34, mul_35, sub_12
#   out_15 => relu_11
# Graph fragment:
#   %sub_12 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_12, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_12, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_28,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_16', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/xg/cxghp6xx4njyamxjmpn2bsmvlcanlgi75snzo6yb65k743o3fxa7.py
# Topologically Sorted Source Nodes: [input_6, batch_norm_12, out_17], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_12 => add_31, mul_37, mul_38, sub_13
#   input_6 => add_29
#   out_17 => relu_12
# Graph fragment:
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convolution_13), kwargs = {})
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_29, %unsqueeze_97), kwargs = {})
#   %mul_37 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_99), kwargs = {})
#   %mul_38 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_37, %unsqueeze_101), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_38, %unsqueeze_103), kwargs = {})
#   %relu_12 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
#   %sub_50 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_29, %unsqueeze_466), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/f3/cf3er2nqn2ymfre6nk4cyuoyi4726yn2jyuqzbhrlj3ow7a6nwc4.py
# Topologically Sorted Source Nodes: [input_6, input_7, batch_norm_14, out_20], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_14 => add_36, mul_43, mul_44, sub_15
#   input_6 => add_29
#   input_7 => add_34
#   out_20 => relu_14
# Graph fragment:
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convolution_13), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %convolution_16), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_36 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %relu_14 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_36,), kwargs = {})
#   %sub_48 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_34, %unsqueeze_442), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/ag/cagiqgoanvaeckex4xkouktiu5h57wcejrn2t6lwr6344mzcqriq.py
# Topologically Sorted Source Nodes: [input_6, input_7, input_8, batch_norm_16, out_23], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_16 => add_41, mul_49, mul_50, sub_17
#   input_6 => add_29
#   input_7 => add_34
#   input_8 => add_39
#   out_23 => relu_16
# Graph fragment:
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convolution_13), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %convolution_16), kwargs = {})
#   %add_39 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %convolution_18), kwargs = {})
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_39, %unsqueeze_129), kwargs = {})
#   %mul_49 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_131), kwargs = {})
#   %mul_50 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_49, %unsqueeze_133), kwargs = {})
#   %add_41 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_50, %unsqueeze_135), kwargs = {})
#   %relu_16 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_41,), kwargs = {})
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_39, %unsqueeze_418), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 320)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x2), tmp23, None)
    tl.store(out_ptr1 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/je/cjevd3tyvv4l3cicy5nt5bia5fpoqqlvrrwluv27zbtzfpifgw6s.py
# Topologically Sorted Source Nodes: [input_6, input_7, input_8, input_9, batch_norm_18, out_26], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_18 => add_46, mul_55, mul_56, sub_19
#   input_6 => add_29
#   input_7 => add_34
#   input_8 => add_39
#   input_9 => add_44
#   out_26 => relu_18
# Graph fragment:
#   %add_29 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_14, %convolution_13), kwargs = {})
#   %add_34 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_29, %convolution_16), kwargs = {})
#   %add_39 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_34, %convolution_18), kwargs = {})
#   %add_44 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_39, %convolution_20), kwargs = {})
#   %sub_19 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_44, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_19, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
#   %relu_18 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_46,), kwargs = {})
#   %sub_44 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_44, %unsqueeze_394), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1310720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 320)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp25, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/to/ctogj7vkdqd4cp6mun62nxcj7wldarlwjuubsanvgp5y7dnysck5.py
# Topologically Sorted Source Nodes: [batch_norm_21, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   batch_norm_21 => add_53, mul_64, mul_65, sub_22
#   out_29 => relu_21
# Graph fragment:
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_53 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %relu_21 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_53,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_21', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = libdevice.sqrt(tmp5)
    tmp7 = tl.full([1], 1, tl.int32)
    tmp8 = tmp7 / tmp6
    tmp9 = 1.0
    tmp10 = tmp8 * tmp9
    tmp11 = tmp2 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: inductor_cache/l5/cl5p2le27a7vvsjd23d2dmwdn6ocbof2zu7nxicvrnktmtflijak.py
# Topologically Sorted Source Nodes: [input_11, batch_norm_22, out_31], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_22 => add_56, mul_67, mul_68, sub_23
#   input_11 => add_54
#   out_31 => relu_22
# Graph fragment:
#   %add_54 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convolution_24), kwargs = {})
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_54, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_56 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %relu_22 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_56,), kwargs = {})
#   %sub_40 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_54, %unsqueeze_346), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.sqrt(tmp7)
    tmp9 = tl.full([1], 1, tl.int32)
    tmp10 = tmp9 / tmp8
    tmp11 = 1.0
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(out_ptr0 + (x2), tmp19, None)
    tl.store(out_ptr1 + (x2), tmp4, None)
''', device_str='cuda')


# kernel path: inductor_cache/ny/cnyj6h6zt7xdtopfowcvh2oh4nvaeu6xxl5axfnjrhlftbflmqmn.py
# Topologically Sorted Source Nodes: [input_11, input_12, batch_norm_24, out_34], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_24 => add_61, mul_73, mul_74, sub_25
#   input_11 => add_54
#   input_12 => add_59
#   out_34 => relu_24
# Graph fragment:
#   %add_54 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convolution_24), kwargs = {})
#   %add_59 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %convolution_27), kwargs = {})
#   %sub_25 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_59, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_25, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_24 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_61,), kwargs = {})
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_59, %unsqueeze_322), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 1e-05
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.sqrt(tmp9)
    tmp11 = tl.full([1], 1, tl.int32)
    tmp12 = tmp11 / tmp10
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp6 * tmp14
    tmp17 = tmp15 * tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0, tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp19)
    tl.store(out_ptr0 + (x2), tmp21, None)
    tl.store(out_ptr1 + (x2), tmp6, None)
''', device_str='cuda')


# kernel path: inductor_cache/7v/c7v2lrf4yyk62m5qcdarf47izerbfsbo6q32kjg5s5wssohyaprz.py
# Topologically Sorted Source Nodes: [input_11, input_12, input_13, batch_norm_26, out_37], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_26 => add_66, mul_79, mul_80, sub_27
#   input_11 => add_54
#   input_12 => add_59
#   input_13 => add_64
#   out_37 => relu_26
# Graph fragment:
#   %add_54 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convolution_24), kwargs = {})
#   %add_59 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %convolution_27), kwargs = {})
#   %add_64 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_59, %convolution_29), kwargs = {})
#   %sub_27 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_64, %unsqueeze_209), kwargs = {})
#   %mul_79 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_27, %unsqueeze_211), kwargs = {})
#   %mul_80 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_79, %unsqueeze_213), kwargs = {})
#   %add_66 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_80, %unsqueeze_215), kwargs = {})
#   %relu_26 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_66,), kwargs = {})
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_64, %unsqueeze_298), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_24', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 640)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x2), None)
    tmp3 = tl.load(in_ptr2 + (x2), None)
    tmp5 = tl.load(in_ptr3 + (x2), None)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.sqrt(tmp11)
    tmp13 = tl.full([1], 1, tl.int32)
    tmp14 = tmp13 / tmp12
    tmp15 = 1.0
    tmp16 = tmp14 * tmp15
    tmp17 = tmp8 * tmp16
    tmp19 = tmp17 * tmp18
    tmp21 = tmp19 + tmp20
    tmp22 = tl.full([1], 0, tl.int32)
    tmp23 = triton_helpers.maximum(tmp22, tmp21)
    tl.store(out_ptr0 + (x2), tmp23, None)
    tl.store(out_ptr1 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: inductor_cache/tb/ctbtkq63vbgsr4rq2233jeokvf7yuyii776pmutnmqjtbeygc5rt.py
# Topologically Sorted Source Nodes: [input_11, input_12, input_13, input_14, batch_norm_28, out_40], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
# Source node to ATen node mapping:
#   batch_norm_28 => add_71, mul_85, mul_86, sub_29
#   input_11 => add_54
#   input_12 => add_59
#   input_13 => add_64
#   input_14 => add_69
#   out_40 => relu_28
# Graph fragment:
#   %add_54 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_25, %convolution_24), kwargs = {})
#   %add_59 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_54, %convolution_27), kwargs = {})
#   %add_64 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_59, %convolution_29), kwargs = {})
#   %add_69 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_64, %convolution_31), kwargs = {})
#   %sub_29 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_69, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_29, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_71 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %relu_28 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_71,), kwargs = {})
#   %sub_34 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_69, %unsqueeze_274), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_25', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 655360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    x1 = (xindex % 640)
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.load(in_ptr0 + (x0), None)
    tmp3 = tl.load(in_ptr1 + (x0), None)
    tmp5 = tl.load(in_ptr2 + (x0), None)
    tmp7 = tl.load(in_ptr3 + (x0), None)
    tmp9 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.sqrt(tmp13)
    tmp15 = tl.full([1], 1, tl.int32)
    tmp16 = tmp15 / tmp14
    tmp17 = 1.0
    tmp18 = tmp16 * tmp17
    tmp19 = tmp10 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tmp24 = tl.full([1], 0, tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp23)
    tl.store(in_out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr0 + (x0), tmp25, None)
    tl.store(out_ptr1 + (x0), tmp10, None)
''', device_str='cuda')


# kernel path: inductor_cache/ii/ciijbl6lhwuqgu6ci2z54gxftleda4v2bhb3oxrchibp5ewbv43r.py
# Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.view]
# Source node to ATen node mapping:
#   out_45 => view
# Graph fragment:
#   %view : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%avg_pool2d, [-1, 640]), kwargs = {})
triton_poi_fused_view_26 = async_compile.triton('triton_poi_fused_view_26', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_view_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 640)
    x1 = xindex // 640
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (640*((x0 % 4)) + 2560*((x0 + 640*x1) // 2560) + ((((x0 + 640*x1) // 4) % 640))), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163 = args
    args.clear()
    assert_size_stride(primals_1, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (1, 3, 1, 1), (3, 1, 1, 1))
    assert_size_stride(primals_4, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, ), (1, ))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (160, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_10, (160, ), (1, ))
    assert_size_stride(primals_11, (160, ), (1, ))
    assert_size_stride(primals_12, (160, ), (1, ))
    assert_size_stride(primals_13, (160, ), (1, ))
    assert_size_stride(primals_14, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_15, (160, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_16, (160, ), (1, ))
    assert_size_stride(primals_17, (160, ), (1, ))
    assert_size_stride(primals_18, (160, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_20, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_24, (160, ), (1, ))
    assert_size_stride(primals_25, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_26, (160, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_28, (160, ), (1, ))
    assert_size_stride(primals_29, (160, ), (1, ))
    assert_size_stride(primals_30, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_31, (160, ), (1, ))
    assert_size_stride(primals_32, (160, ), (1, ))
    assert_size_stride(primals_33, (160, ), (1, ))
    assert_size_stride(primals_34, (160, ), (1, ))
    assert_size_stride(primals_35, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_36, (160, ), (1, ))
    assert_size_stride(primals_37, (160, ), (1, ))
    assert_size_stride(primals_38, (160, ), (1, ))
    assert_size_stride(primals_39, (160, ), (1, ))
    assert_size_stride(primals_40, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_41, (160, ), (1, ))
    assert_size_stride(primals_42, (160, ), (1, ))
    assert_size_stride(primals_43, (160, ), (1, ))
    assert_size_stride(primals_44, (160, ), (1, ))
    assert_size_stride(primals_45, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_46, (160, ), (1, ))
    assert_size_stride(primals_47, (160, ), (1, ))
    assert_size_stride(primals_48, (160, ), (1, ))
    assert_size_stride(primals_49, (160, ), (1, ))
    assert_size_stride(primals_50, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_51, (160, ), (1, ))
    assert_size_stride(primals_52, (160, ), (1, ))
    assert_size_stride(primals_53, (160, ), (1, ))
    assert_size_stride(primals_54, (160, ), (1, ))
    assert_size_stride(primals_55, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_56, (160, ), (1, ))
    assert_size_stride(primals_57, (160, ), (1, ))
    assert_size_stride(primals_58, (160, ), (1, ))
    assert_size_stride(primals_59, (160, ), (1, ))
    assert_size_stride(primals_60, (320, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_61, (320, ), (1, ))
    assert_size_stride(primals_62, (320, ), (1, ))
    assert_size_stride(primals_63, (320, ), (1, ))
    assert_size_stride(primals_64, (320, ), (1, ))
    assert_size_stride(primals_65, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_66, (320, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_67, (320, ), (1, ))
    assert_size_stride(primals_68, (320, ), (1, ))
    assert_size_stride(primals_69, (320, ), (1, ))
    assert_size_stride(primals_70, (320, ), (1, ))
    assert_size_stride(primals_71, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_72, (320, ), (1, ))
    assert_size_stride(primals_73, (320, ), (1, ))
    assert_size_stride(primals_74, (320, ), (1, ))
    assert_size_stride(primals_75, (320, ), (1, ))
    assert_size_stride(primals_76, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_77, (320, ), (1, ))
    assert_size_stride(primals_78, (320, ), (1, ))
    assert_size_stride(primals_79, (320, ), (1, ))
    assert_size_stride(primals_80, (320, ), (1, ))
    assert_size_stride(primals_81, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_82, (320, ), (1, ))
    assert_size_stride(primals_83, (320, ), (1, ))
    assert_size_stride(primals_84, (320, ), (1, ))
    assert_size_stride(primals_85, (320, ), (1, ))
    assert_size_stride(primals_86, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_87, (320, ), (1, ))
    assert_size_stride(primals_88, (320, ), (1, ))
    assert_size_stride(primals_89, (320, ), (1, ))
    assert_size_stride(primals_90, (320, ), (1, ))
    assert_size_stride(primals_91, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_92, (320, ), (1, ))
    assert_size_stride(primals_93, (320, ), (1, ))
    assert_size_stride(primals_94, (320, ), (1, ))
    assert_size_stride(primals_95, (320, ), (1, ))
    assert_size_stride(primals_96, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_97, (320, ), (1, ))
    assert_size_stride(primals_98, (320, ), (1, ))
    assert_size_stride(primals_99, (320, ), (1, ))
    assert_size_stride(primals_100, (320, ), (1, ))
    assert_size_stride(primals_101, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_102, (320, ), (1, ))
    assert_size_stride(primals_103, (320, ), (1, ))
    assert_size_stride(primals_104, (320, ), (1, ))
    assert_size_stride(primals_105, (320, ), (1, ))
    assert_size_stride(primals_106, (320, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_107, (320, ), (1, ))
    assert_size_stride(primals_108, (320, ), (1, ))
    assert_size_stride(primals_109, (320, ), (1, ))
    assert_size_stride(primals_110, (320, ), (1, ))
    assert_size_stride(primals_111, (640, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_112, (640, ), (1, ))
    assert_size_stride(primals_113, (640, ), (1, ))
    assert_size_stride(primals_114, (640, ), (1, ))
    assert_size_stride(primals_115, (640, ), (1, ))
    assert_size_stride(primals_116, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_117, (640, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_118, (640, ), (1, ))
    assert_size_stride(primals_119, (640, ), (1, ))
    assert_size_stride(primals_120, (640, ), (1, ))
    assert_size_stride(primals_121, (640, ), (1, ))
    assert_size_stride(primals_122, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_123, (640, ), (1, ))
    assert_size_stride(primals_124, (640, ), (1, ))
    assert_size_stride(primals_125, (640, ), (1, ))
    assert_size_stride(primals_126, (640, ), (1, ))
    assert_size_stride(primals_127, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_128, (640, ), (1, ))
    assert_size_stride(primals_129, (640, ), (1, ))
    assert_size_stride(primals_130, (640, ), (1, ))
    assert_size_stride(primals_131, (640, ), (1, ))
    assert_size_stride(primals_132, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_133, (640, ), (1, ))
    assert_size_stride(primals_134, (640, ), (1, ))
    assert_size_stride(primals_135, (640, ), (1, ))
    assert_size_stride(primals_136, (640, ), (1, ))
    assert_size_stride(primals_137, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_138, (640, ), (1, ))
    assert_size_stride(primals_139, (640, ), (1, ))
    assert_size_stride(primals_140, (640, ), (1, ))
    assert_size_stride(primals_141, (640, ), (1, ))
    assert_size_stride(primals_142, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_143, (640, ), (1, ))
    assert_size_stride(primals_144, (640, ), (1, ))
    assert_size_stride(primals_145, (640, ), (1, ))
    assert_size_stride(primals_146, (640, ), (1, ))
    assert_size_stride(primals_147, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_148, (640, ), (1, ))
    assert_size_stride(primals_149, (640, ), (1, ))
    assert_size_stride(primals_150, (640, ), (1, ))
    assert_size_stride(primals_151, (640, ), (1, ))
    assert_size_stride(primals_152, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_153, (640, ), (1, ))
    assert_size_stride(primals_154, (640, ), (1, ))
    assert_size_stride(primals_155, (640, ), (1, ))
    assert_size_stride(primals_156, (640, ), (1, ))
    assert_size_stride(primals_157, (640, 640, 3, 3), (5760, 9, 3, 1))
    assert_size_stride(primals_158, (640, ), (1, ))
    assert_size_stride(primals_159, (640, ), (1, ))
    assert_size_stride(primals_160, (640, ), (1, ))
    assert_size_stride(primals_161, (640, ), (1, ))
    assert_size_stride(primals_162, (100, 640), (640, 1))
    assert_size_stride(primals_163, (100, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_4, buf0, 48, 9, grid=grid(48, 9), stream=stream0)
        del primals_4
        buf1 = empty_strided_cuda((160, 16, 3, 3), (144, 1, 48, 16), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_9, buf1, 2560, 9, grid=grid(2560, 9), stream=stream0)
        del primals_9
        buf31 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Topologically Sorted Source Nodes: [sub, x], Original ATen: [aten.sub, aten.div]
        stream0 = get_raw_stream(0)
        triton_poi_fused_div_sub_2.run(primals_2, primals_1, primals_3, buf31, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_1
        del primals_2
        del primals_3
        buf2 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_14, buf2, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_14
        buf3 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_20, buf3, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_20
        buf4 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_25, buf4, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_25
        buf5 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_30, buf5, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_30
        buf6 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_35, buf6, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_35
        buf7 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_40, buf7, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_40
        buf8 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_45, buf8, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_45
        buf9 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_50, buf9, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_50
        buf10 = empty_strided_cuda((160, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_55, buf10, 25600, 9, grid=grid(25600, 9), stream=stream0)
        del primals_55
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, buf0, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 16, 64, 64), (65536, 1, 1024, 16))
        buf33 = empty_strided_cuda((4, 16, 64, 64), (65536, 1, 1024, 16), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf32, primals_5, primals_6, primals_7, primals_8, buf33, 262144, grid=grid(262144), stream=stream0)
        del primals_8
        buf11 = empty_strided_cuda((320, 160, 3, 3), (1440, 1, 480, 160), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_60, buf11, 51200, 9, grid=grid(51200, 9), stream=stream0)
        del primals_60
        buf12 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_65, buf12, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_65
        buf13 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_71, buf13, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_71
        buf14 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_76, buf14, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_76
        buf15 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_81, buf15, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_81
        buf16 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_86, buf16, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_86
        buf17 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_91, buf17, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_91
        buf18 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_96, buf18, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_96
        buf19 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_101, buf19, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_101
        buf20 = empty_strided_cuda((320, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_6.run(primals_106, buf20, 102400, 9, grid=grid(102400, 9), stream=stream0)
        del primals_106
        buf21 = empty_strided_cuda((640, 320, 3, 3), (2880, 1, 960, 320), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_7.run(primals_111, buf21, 204800, 9, grid=grid(204800, 9), stream=stream0)
        del primals_111
        # Topologically Sorted Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf35 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf34, primals_10, primals_11, primals_12, primals_13, buf35, 2621440, grid=grid(2621440), stream=stream0)
        del primals_13
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 160, 64, 64), (655360, 1, 10240, 160))
        # Topologically Sorted Source Nodes: [conv2d_3], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf33, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf38 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, batch_norm_2, out_3], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf37, buf36, primals_16, primals_17, primals_18, primals_19, buf38, 2621440, grid=grid(2621440), stream=stream0)
        del primals_19
        # Topologically Sorted Source Nodes: [conv2d_4], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf40 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_3, out_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf39, primals_21, primals_22, primals_23, primals_24, buf40, 2621440, grid=grid(2621440), stream=stream0)
        del primals_24
        # Topologically Sorted Source Nodes: [out_5], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf42 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, batch_norm_4, out_6], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10.run(buf37, buf36, buf41, primals_26, primals_27, primals_28, primals_29, buf42, 2621440, grid=grid(2621440), stream=stream0)
        del primals_29
        # Topologically Sorted Source Nodes: [conv2d_6], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf44 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_5, out_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf43, primals_31, primals_32, primals_33, primals_34, buf44, 2621440, grid=grid(2621440), stream=stream0)
        del primals_34
        # Topologically Sorted Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf46 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, batch_norm_6, out_9], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf37, buf36, buf41, buf45, primals_36, primals_37, primals_38, primals_39, buf46, 2621440, grid=grid(2621440), stream=stream0)
        del primals_39
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf48 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_7, out_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf47, primals_41, primals_42, primals_43, primals_44, buf48, 2621440, grid=grid(2621440), stream=stream0)
        del primals_44
        # Topologically Sorted Source Nodes: [out_11], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf22 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_116, buf22, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_116
        buf23 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_122, buf23, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_122
        buf24 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_127, buf24, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_127
        buf25 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_132, buf25, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_132
        buf26 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_137, buf26, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_137
        buf27 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_142, buf27, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_142
        buf28 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_147, buf28, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_147
        buf29 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_152, buf29, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_152
        buf30 = empty_strided_cuda((640, 640, 3, 3), (5760, 1, 1920, 640), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_12.run(primals_157, buf30, 409600, 9, grid=grid(409600, 9), stream=stream0)
        del primals_157
        buf50 = buf49; del buf49  # reuse
        buf51 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        buf115 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3, input_4, batch_norm_8, out_12], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_13.run(buf50, buf37, buf36, buf41, buf45, primals_46, primals_47, primals_48, primals_49, buf51, buf115, 2621440, grid=grid(2621440), stream=stream0)
        del primals_46
        del primals_49
        buf116 = buf45; del buf45  # reuse
        buf117 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        buf118 = empty_strided_cuda((4, 160, 64, 64), (655360, 1, 10240, 160), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2, input_3], Original ATen: [aten.add, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_native_batch_norm_backward_14.run(buf116, buf37, buf36, buf41, primals_36, primals_26, primals_16, buf117, buf118, 2621440, grid=grid(2621440), stream=stream0)
        del primals_16
        del primals_26
        del primals_36
        # Topologically Sorted Source Nodes: [conv2d_10], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf53 = buf41; del buf41  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_9, out_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf52, primals_51, primals_52, primals_53, primals_54, buf53, 2621440, grid=grid(2621440), stream=stream0)
        del primals_54
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, buf10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 160, 64, 64), (655360, 1, 10240, 160))
        buf55 = buf37; del buf37  # reuse
        buf114 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [input_5, batch_norm_10, x_2], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_15.run(buf50, buf54, primals_56, primals_57, primals_58, primals_59, buf55, buf114, 2621440, grid=grid(2621440), stream=stream0)
        del buf50
        del buf54
        del primals_56
        del primals_59
        # Topologically Sorted Source Nodes: [conv2d_12], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, buf11, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf57 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_11, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf56, primals_61, primals_62, primals_63, primals_64, buf57, 1310720, grid=grid(1310720), stream=stream0)
        del primals_64
        # Topologically Sorted Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 320, 32, 32), (327680, 1, 10240, 320))
        # Topologically Sorted Source Nodes: [conv2d_14], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf55, primals_66, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf60 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf113 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, batch_norm_12, out_17], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17.run(buf59, buf58, primals_67, primals_68, primals_69, primals_70, buf60, buf113, 1310720, grid=grid(1310720), stream=stream0)
        del primals_67
        del primals_70
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf62 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_13, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf61, primals_72, primals_73, primals_74, primals_75, buf62, 1310720, grid=grid(1310720), stream=stream0)
        del primals_75
        # Topologically Sorted Source Nodes: [out_19], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf64 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf112 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7, batch_norm_14, out_20], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_18.run(buf59, buf58, buf63, primals_77, primals_78, primals_79, primals_80, buf64, buf112, 1310720, grid=grid(1310720), stream=stream0)
        del primals_77
        del primals_80
        # Topologically Sorted Source Nodes: [conv2d_17], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf66 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_15, out_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf65, primals_82, primals_83, primals_84, primals_85, buf66, 1310720, grid=grid(1310720), stream=stream0)
        del primals_85
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, buf16, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf68 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf111 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7, input_8, batch_norm_16, out_23], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_19.run(buf59, buf58, buf63, buf67, primals_87, primals_88, primals_89, primals_90, buf68, buf111, 1310720, grid=grid(1310720), stream=stream0)
        del primals_87
        del primals_90
        # Topologically Sorted Source Nodes: [conv2d_19], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, buf17, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf70 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_17, out_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf69, primals_92, primals_93, primals_94, primals_95, buf70, 1310720, grid=grid(1310720), stream=stream0)
        del primals_95
        # Topologically Sorted Source Nodes: [out_25], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf72 = buf59; del buf59  # reuse
        buf73 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        buf110 = empty_strided_cuda((4, 320, 32, 32), (327680, 1, 10240, 320), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7, input_8, input_9, batch_norm_18, out_26], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_20.run(buf72, buf58, buf63, buf67, buf71, primals_97, primals_98, primals_99, primals_100, buf73, buf110, 1310720, grid=grid(1310720), stream=stream0)
        del buf58
        del primals_100
        del primals_97
        # Topologically Sorted Source Nodes: [conv2d_21], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, buf19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf75 = buf71; del buf71  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_19, out_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf74, primals_102, primals_103, primals_104, primals_105, buf75, 1310720, grid=grid(1310720), stream=stream0)
        del primals_105
        # Topologically Sorted Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 320, 32, 32), (327680, 1, 10240, 320))
        buf77 = buf67; del buf67  # reuse
        buf109 = buf63; del buf63  # reuse
        # Topologically Sorted Source Nodes: [input_10, batch_norm_20, x_3], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_17.run(buf72, buf76, primals_107, primals_108, primals_109, primals_110, buf77, buf109, 1310720, grid=grid(1310720), stream=stream0)
        del buf72
        del buf76
        del primals_107
        del primals_110
        # Topologically Sorted Source Nodes: [conv2d_23], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, buf21, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf79 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_21, out_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf78, primals_112, primals_113, primals_114, primals_115, buf79, 655360, grid=grid(655360), stream=stream0)
        del primals_115
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, buf22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 640, 16, 16), (163840, 1, 10240, 640))
        # Topologically Sorted Source Nodes: [conv2d_25], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf77, primals_117, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf82 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf108 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, batch_norm_22, out_31], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22.run(buf81, buf80, primals_118, primals_119, primals_120, primals_121, buf82, buf108, 655360, grid=grid(655360), stream=stream0)
        del primals_118
        del primals_121
        # Topologically Sorted Source Nodes: [conv2d_26], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, buf23, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf84 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_23, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf83, primals_123, primals_124, primals_125, primals_126, buf84, 655360, grid=grid(655360), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf86 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf107 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, batch_norm_24, out_34], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_23.run(buf81, buf80, buf85, primals_128, primals_129, primals_130, primals_131, buf86, buf107, 655360, grid=grid(655360), stream=stream0)
        del primals_128
        del primals_131
        # Topologically Sorted Source Nodes: [conv2d_28], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf88 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_25, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf87, primals_133, primals_134, primals_135, primals_136, buf88, 655360, grid=grid(655360), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, buf26, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf90 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf106 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13, batch_norm_26, out_37], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_24.run(buf81, buf80, buf85, buf89, primals_138, primals_139, primals_140, primals_141, buf90, buf106, 655360, grid=grid(655360), stream=stream0)
        del primals_138
        del primals_141
        # Topologically Sorted Source Nodes: [conv2d_30], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf92 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [batch_norm_27, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf91, primals_143, primals_144, primals_145, primals_146, buf92, 655360, grid=grid(655360), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [out_39], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf94 = buf81; del buf81  # reuse
        buf95 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        buf105 = empty_strided_cuda((4, 640, 16, 16), (163840, 1, 10240, 640), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12, input_13, input_14, batch_norm_28, out_40], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_25.run(buf94, buf80, buf85, buf89, buf93, primals_148, primals_149, primals_150, primals_151, buf95, buf105, 655360, grid=grid(655360), stream=stream0)
        del buf80
        del primals_148
        del primals_151
        # Topologically Sorted Source Nodes: [conv2d_32], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, buf29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf97 = buf93; del buf93  # reuse
        # Topologically Sorted Source Nodes: [batch_norm_29, out_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf96, primals_153, primals_154, primals_155, primals_156, buf97, 655360, grid=grid(655360), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, buf30, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 640, 16, 16), (163840, 1, 10240, 640))
        buf99 = buf89; del buf89  # reuse
        buf104 = buf85; del buf85  # reuse
        # Topologically Sorted Source Nodes: [input_15, batch_norm_30, out_43], Original ATen: [aten.add, aten._native_batch_norm_legit_no_training, aten.relu, aten.native_batch_norm_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_native_batch_norm_backward_relu_22.run(buf94, buf98, primals_158, primals_159, primals_160, primals_161, buf99, buf104, 655360, grid=grid(655360), stream=stream0)
        del buf94
        del buf98
        del primals_158
        del primals_161
        # Topologically Sorted Source Nodes: [out_44], Original ATen: [aten.avg_pool2d]
        buf100 = torch.ops.aten.avg_pool2d.default(buf99, [8, 8], [8, 8], [0, 0], False, True, None)
        buf101 = buf100
        del buf100
        buf102 = empty_strided_cuda((16, 640), (640, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out_45], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused_view_26.run(buf101, buf102, 10240, grid=grid(10240), stream=stream0)
        del buf101
        buf103 = empty_strided_cuda((16, 100), (100, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_163, buf102, reinterpret_tensor(primals_162, (640, 100), (1, 640), 0), alpha=1, beta=1, out=buf103)
        del primals_163
    return (buf103, buf0, primals_5, primals_6, primals_7, buf1, primals_10, primals_11, primals_12, buf2, primals_15, primals_17, primals_18, buf3, primals_21, primals_22, primals_23, buf4, primals_27, primals_28, buf5, primals_31, primals_32, primals_33, buf6, primals_37, primals_38, buf7, primals_41, primals_42, primals_43, buf8, primals_47, primals_48, buf9, primals_51, primals_52, primals_53, buf10, primals_57, primals_58, buf11, primals_61, primals_62, primals_63, buf12, primals_66, primals_68, primals_69, buf13, primals_72, primals_73, primals_74, buf14, primals_78, primals_79, buf15, primals_82, primals_83, primals_84, buf16, primals_88, primals_89, buf17, primals_92, primals_93, primals_94, buf18, primals_98, primals_99, buf19, primals_102, primals_103, primals_104, buf20, primals_108, primals_109, buf21, primals_112, primals_113, primals_114, buf22, primals_117, primals_119, primals_120, buf23, primals_123, primals_124, primals_125, buf24, primals_129, primals_130, buf25, primals_133, primals_134, primals_135, buf26, primals_139, primals_140, buf27, primals_143, primals_144, primals_145, buf28, primals_149, primals_150, buf29, primals_153, primals_154, primals_155, buf30, primals_159, primals_160, buf31, buf32, buf33, buf34, buf35, buf38, buf39, buf40, buf42, buf43, buf44, buf46, buf47, buf48, buf51, buf52, buf53, buf55, buf56, buf57, buf60, buf61, buf62, buf64, buf65, buf66, buf68, buf69, buf70, buf73, buf74, buf75, buf77, buf78, buf79, buf82, buf83, buf84, buf86, buf87, buf88, buf90, buf91, buf92, buf95, buf96, buf97, buf99, buf102, primals_162, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((1, 3, 1, 1), (3, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((160, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((160, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((320, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((320, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((320, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((640, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((640, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((640, 640, 3, 3), (5760, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((100, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((100, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
