# AOT ID: ['9_forward']
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


# kernel path: inductor_cache/dq/cdq3h7u7hbia7enqve2p6q7tnmwfo7ieewqjxqkuspsdo3kpimq7.py
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
    size_hints={'y': 16, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16
    xnumel = 16
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
    tmp0 = tl.load(in_ptr0 + (x2 + 16*y3), xmask & ymask)
    tl.store(out_ptr0 + (y0 + 4*x2 + 64*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/kf/ckfdvpgo7vdfunzsnubeniqwhis2ppgujxz4ybqx66k22psrgppg.py
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
    size_hints={'y': 64, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 4*x2 + 36*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/5r/c5rnmif6cvouutzbqeqt5lh4zv2gdiv2svvyiaidtdd4wg25szlr.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8576
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 67)
    y1 = yindex // 67
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 67*x2 + 603*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/xq/cxq5x7vmjo4ozks6lvujw65xbprv5sp5q5ykuknzpjs3uyptnzsf.py
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
    size_hints={'y': 16384, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/qt/cqtt4hx7oedoeeoexc2kt23klw5vhg2ljkiqrpq6xvs2nwwtjhyo.py
# Unsorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 33536
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 131)
    y1 = yindex // 131
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 131*x2 + 1179*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/jz/cjzycjbnaysn6ceizsvo7tdnjyzuomtjqxp4vstukmfig4fje5oa.py
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
    size_hints={'y': 131072, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 49*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 256*x2 + 12544*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/bt/cbtwpipiygj7a6vucs7fapvun5wjt7jirmizubp3tzyrvhecqszs.py
# Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_2 => add_1, mul_1, mul_2, sub
#   input_3 => clamp_max, clamp_min
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_1, 0.0), kwargs = {})
#   %clamp_max : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 16)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/x2/cx2h66rulvfyta5ecvuz6sofjz2ymbm7euetp5otyoaghurhmxzs.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_8 => add_5, mul_7, mul_8, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/eh/cehq5ip7x5wjxcl55grojzvbvcq2x5lgpqczrkn4hzp7fxls64d7.py
# Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_10 => add_7, mul_10, mul_11, sub_3
#   input_11 => clamp_max_2, clamp_min_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %clamp_min_2 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_7, 0.0), kwargs = {})
#   %clamp_max_2 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_2, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zg/czgtmvkym7iysbfmpiq2zh4tnjoq6ain6nxrwzk5ow2zcivzkd4n.py
# Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_13 => add_9, mul_13, mul_14, sub_4
#   input_14 => clamp_max_3, clamp_min_3
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %clamp_min_3 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_9, 0.0), kwargs = {})
#   %clamp_max_3 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_3, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ze/czeeeuqrdxkunue6l2dahpowpda763opntp3sq4zavklyzpxfooz.py
# Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_16 => add_11, mul_16, mul_17, sub_5
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_11 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_10', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/q4/cq46j24qoulfoetgl6wzpfdoq7jytfdswkixw3trnkijmqnpqve6.py
# Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_24 => add_17, mul_25, mul_26, sub_8
#   input_25 => add_18
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_18 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_11, %add_17), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_11', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1024}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ai/cai2notzigrs7ywqmy2huozpl7qgk3t3ajkargsike4yxhuiaioy.py
# Topologically Sorted Source Nodes: [input_60], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_60 => add_45, mul_61, mul_62, sub_20
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_45 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_12', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ar/cara35i2sa3nabzk6von577eh44nn2alhlowl24yepstuy23ujzk.py
# Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_62 => add_47, mul_64, mul_65, sub_21
#   input_63 => clamp_max_14, clamp_min_14
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %clamp_min_14 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_47, 0.0), kwargs = {})
#   %clamp_max_14 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_14, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 1536)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zk/czkfbfpt2lvy33za3hij57qb44wr6jawnolsyb6mn4cydvwtd4z4.py
# Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_68 => add_51, mul_70, mul_71, sub_23
#   input_69 => add_52
# Graph fragment:
#   %sub_23 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_23, %unsqueeze_185), kwargs = {})
#   %mul_70 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_23, %unsqueeze_187), kwargs = {})
#   %mul_71 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_70, %unsqueeze_189), kwargs = {})
#   %add_51 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_71, %unsqueeze_191), kwargs = {})
#   %add_52 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %add_51), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_14', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 384)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1e-05
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = tl.full([1], 1, tl.int32)
    tmp9 = tmp8 / tmp7
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp0 + tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/s7/cs7dn2jdgv4xr63dhk5uyzk6g2undxzgc6ynqsnkj5cg76lxgype.py
# Topologically Sorted Source Nodes: [input_116, input_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_116 => add_89, mul_118, mul_119, sub_39
#   input_117 => clamp_max_26, clamp_min_26
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_39, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %clamp_min_26 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_89, 0.0), kwargs = {})
#   %clamp_max_26 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_26, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i4/ci4o3jnbq5obtmsciucbxhpbdmkvrpkaetevjq5jtgxsso77d2ai.py
# Topologically Sorted Source Nodes: [input_122], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_122 => add_93, mul_124, mul_125, sub_41
# Graph fragment:
#   %sub_41 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_41, %unsqueeze_329), kwargs = {})
#   %mul_124 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_41, %unsqueeze_331), kwargs = {})
#   %mul_125 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_124, %unsqueeze_333), kwargs = {})
#   %add_93 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_125, %unsqueeze_335), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/7y/c7yxjjf2wahezbver5ysvrgjv5tuafxxahticz6ja3wriuch5xcj.py
# Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   ret_1 => cat_1
# Graph fragment:
#   %cat_1 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat, %sqrt_42], 1), kwargs = {})
triton_poi_fused_cat_17 = async_compile.triton('triton_poi_fused_cat_17', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 52528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 67)
    x3 = xindex // 13132
    x2 = ((xindex // 938) % 14)
    x1 = ((xindex // 67) % 14)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 66, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 64, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tl.full([1], 1, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = tmp13 & tmp13
    tmp15 = tmp14 & tmp10
    tmp16 = tl.load(in_ptr0 + (64*x3 + (x0)), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = 1.0
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp15, tmp17, tmp18)
    tmp20 = tmp16 / tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp10, tmp20, tmp21)
    tmp23 = tmp5 >= tmp8
    tmp24 = tl.full([1], 65, tl.int64)
    tmp25 = tmp5 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tmp26 & tmp4
    tmp28 = x2
    tmp29 = tmp28.to(tl.float32)
    tmp30 = 0.07692307692307693
    tmp31 = tmp29 * tmp30
    tmp32 = 2.0
    tmp33 = tmp31 * tmp32
    tmp34 = 1.0
    tmp35 = tmp33 - tmp34
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp27, tmp35, tmp36)
    tmp38 = tmp5 >= tmp24
    tmp39 = tl.full([1], 66, tl.int64)
    tmp40 = tmp5 < tmp39
    tmp41 = tmp38 & tmp4
    tmp42 = x1
    tmp43 = tmp42.to(tl.float32)
    tmp44 = 0.07692307692307693
    tmp45 = tmp43 * tmp44
    tmp46 = 2.0
    tmp47 = tmp45 * tmp46
    tmp48 = 1.0
    tmp49 = tmp47 - tmp48
    tmp50 = tl.full(tmp49.shape, 0.0, tmp49.dtype)
    tmp51 = tl.where(tmp41, tmp49, tmp50)
    tmp52 = tl.where(tmp26, tmp37, tmp51)
    tmp53 = tl.where(tmp9, tmp22, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp4, tmp53, tmp54)
    tmp56 = tmp0 >= tmp3
    tmp57 = tl.full([1], 67, tl.int64)
    tmp58 = tmp0 < tmp57
    tmp59 = x2
    tmp60 = tmp59.to(tl.float32)
    tmp61 = 0.07692307692307693
    tmp62 = tmp60 * tmp61
    tmp63 = 2.0
    tmp64 = tmp62 * tmp63
    tmp65 = 1.0
    tmp66 = tmp64 - tmp65
    tmp67 = 0.5
    tmp68 = tmp66 - tmp67
    tmp69 = tmp68 * tmp68
    tmp70 = x1
    tmp71 = tmp70.to(tl.float32)
    tmp72 = tmp71 * tmp61
    tmp73 = tmp72 * tmp63
    tmp74 = tmp73 - tmp65
    tmp75 = tmp74 - tmp67
    tmp76 = tmp75 * tmp75
    tmp77 = tmp69 + tmp76
    tmp78 = libdevice.sqrt(tmp77)
    tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
    tmp80 = tl.where(tmp56, tmp78, tmp79)
    tmp81 = tl.where(tmp4, tmp55, tmp80)
    tl.store(out_ptr0 + (x6), tmp81, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/jm/cjm3eo5y2e3kuekbrkf6e2jylyzdh7qewlac77hv2qlitc3wg6eq.py
# Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   ret_2 => convolution_42
# Graph fragment:
#   %convolution_42 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_1, %primals_212, %primals_213, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_18 = async_compile.triton('triton_poi_fused_convolution_18', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_18(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fu/cfuhactnulw7lin2apvv3zcxfs5mboutnywdc65qd3hewboijk6a.py
# Topologically Sorted Source Nodes: [input_125, input_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
# Source node to ATen node mapping:
#   input_125 => add_96, mul_129, mul_130, sub_46
#   input_126 => clamp_max_28, clamp_min_28
# Graph fragment:
#   %sub_46 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_43, %unsqueeze_337), kwargs = {})
#   %mul_129 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_46, %unsqueeze_339), kwargs = {})
#   %mul_130 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_129, %unsqueeze_341), kwargs = {})
#   %add_96 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_130, %unsqueeze_343), kwargs = {})
#   %clamp_min_28 : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%add_96, 0.0), kwargs = {})
#   %clamp_max_28 : [num_users=2] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min_28, 6.0), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_19', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 6272*y1), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tl.store(out_ptr0 + (x2 + 49*y3), tmp19, xmask & ymask)
''', device_str='cuda')


# kernel path: inductor_cache/ih/cihqbkcc2bwdzr6lk4tajhkfv55kg63hv3n5p6qnsviextw2km4z.py
# Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   ret_4 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%cat_2, %sqrt_44], 1), kwargs = {})
triton_poi_fused_cat_20 = async_compile.triton('triton_poi_fused_cat_20', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25676
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 131)
    x3 = xindex // 6419
    x5 = ((xindex // 131) % 49)
    x2 = ((xindex // 917) % 7)
    x1 = ((xindex // 131) % 7)
    x6 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 130, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = x0
    tmp6 = tl.full([1], 0, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = tl.full([1], 128, tl.int64)
    tmp9 = tmp5 < tmp8
    tmp10 = tmp9 & tmp4
    tmp11 = tl.load(in_ptr0 + (x5 + 49*(x0) + 6272*x3), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tmp5 >= tmp8
    tmp13 = tl.full([1], 129, tl.int64)
    tmp14 = tmp5 < tmp13
    tmp15 = tmp12 & tmp14
    tmp16 = tmp15 & tmp4
    tmp17 = x2
    tmp18 = tmp17.to(tl.float32)
    tmp19 = 0.16666666666666666
    tmp20 = tmp18 * tmp19
    tmp21 = 2.0
    tmp22 = tmp20 * tmp21
    tmp23 = 1.0
    tmp24 = tmp22 - tmp23
    tmp25 = tl.full(tmp24.shape, 0.0, tmp24.dtype)
    tmp26 = tl.where(tmp16, tmp24, tmp25)
    tmp27 = tmp5 >= tmp13
    tmp28 = tl.full([1], 130, tl.int64)
    tmp29 = tmp5 < tmp28
    tmp30 = tmp27 & tmp4
    tmp31 = x1
    tmp32 = tmp31.to(tl.float32)
    tmp33 = 0.16666666666666666
    tmp34 = tmp32 * tmp33
    tmp35 = 2.0
    tmp36 = tmp34 * tmp35
    tmp37 = 1.0
    tmp38 = tmp36 - tmp37
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp30, tmp38, tmp39)
    tmp41 = tl.where(tmp15, tmp26, tmp40)
    tmp42 = tl.where(tmp9, tmp11, tmp41)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp4, tmp42, tmp43)
    tmp45 = tmp0 >= tmp3
    tmp46 = tl.full([1], 131, tl.int64)
    tmp47 = tmp0 < tmp46
    tmp48 = x2
    tmp49 = tmp48.to(tl.float32)
    tmp50 = 0.16666666666666666
    tmp51 = tmp49 * tmp50
    tmp52 = 2.0
    tmp53 = tmp51 * tmp52
    tmp54 = 1.0
    tmp55 = tmp53 - tmp54
    tmp56 = 0.5
    tmp57 = tmp55 - tmp56
    tmp58 = tmp57 * tmp57
    tmp59 = x1
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 * tmp50
    tmp62 = tmp61 * tmp52
    tmp63 = tmp62 - tmp54
    tmp64 = tmp63 - tmp56
    tmp65 = tmp64 * tmp64
    tmp66 = tmp58 + tmp65
    tmp67 = libdevice.sqrt(tmp66)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp45, tmp67, tmp68)
    tmp70 = tl.where(tmp4, tmp44, tmp69)
    tl.store(out_ptr0 + (x6), tmp70, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ju/cjuzxrroi4rnhcj2zotbzsm7sw6uqffa6srhhn2f6iynqdqxft6f.py
# Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   ret_5 => convolution_44
# Graph fragment:
#   %convolution_44 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %primals_219, %primals_220, [1, 1], [1, 1], [1, 1], False, [0, 0], 1), kwargs = {})
triton_poi_fused_convolution_21 = async_compile.triton('triton_poi_fused_convolution_21', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_21(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/uu/cuubzoplxitwce3canav7w5ljnakvyjxwqw3hlu25i5cg3d5bcwk.py
# Topologically Sorted Source Nodes: [tensors], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   tensors => cat_4
# Graph fragment:
#   %cat_4 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%view, %view_3, %view_4], 1), kwargs = {})
triton_poi_fused_cat_22 = async_compile.triton('triton_poi_fused_cat_22', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 77312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 19328)
    x1 = xindex // 19328
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 12544, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = triton_helpers.div_floor_integer((((x0) // 14) % 14),  14)
    tmp6 = 1 + (triton_helpers.div_floor_integer((((x0) // 14) % 14),  14))
    tmp7 = tmp5 < tmp6
    tmp8 = triton_helpers.div_floor_integer(((x0) % 14),  14)
    tmp9 = 1 + (triton_helpers.div_floor_integer(((x0) % 14),  14))
    tmp10 = tmp8 < tmp9
    tmp11 = tmp7 & tmp10
    tmp12 = tmp11 & tmp4
    tmp13 = tl.load(in_ptr0 + (64*x1 + ((((x0) // 196) % 64))), tmp12 & xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = 1.0
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tmp13 / tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp4, tmp17, tmp18)
    tmp20 = tmp0 >= tmp3
    tmp21 = tl.full([1], 18816, tl.int64)
    tmp22 = tmp0 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tl.load(in_ptr1 + (6272*x1 + ((-12544) + x0)), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp0 >= tmp21
    tmp26 = tl.full([1], 19328, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tl.load(in_ptr2 + (512*x1 + ((-18816) + x0)), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 - tmp29
    tmp31 = tl.load(in_ptr4 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = libdevice.sqrt(tmp33)
    tmp35 = tl.full([1], 1, tl.int32)
    tmp36 = tmp35 / tmp34
    tmp37 = 1.0
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp40 = tl.load(in_ptr5 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp39 * tmp40
    tmp42 = tl.load(in_ptr6 + ((-18816) + x0), tmp25 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tmp41 + tmp42
    tmp44 = 0.0
    tmp45 = triton_helpers.maximum(tmp43, tmp44)
    tmp46 = 6.0
    tmp47 = triton_helpers.minimum(tmp45, tmp46)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp25, tmp47, tmp48)
    tmp50 = tl.where(tmp23, tmp24, tmp49)
    tmp51 = tl.where(tmp4, tmp19, tmp50)
    tl.store(out_ptr0 + (x2), tmp51, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227 = args
    args.clear()
    assert_size_stride(primals_1, (4, 4, 4, 4), (64, 16, 4, 1))
    assert_size_stride(primals_2, (16, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, ), (1, ))
    assert_size_stride(primals_6, (16, ), (1, ))
    assert_size_stride(primals_7, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_8, (16, ), (1, ))
    assert_size_stride(primals_9, (16, ), (1, ))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (16, ), (1, ))
    assert_size_stride(primals_12, (192, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_13, (192, ), (1, ))
    assert_size_stride(primals_14, (192, ), (1, ))
    assert_size_stride(primals_15, (192, ), (1, ))
    assert_size_stride(primals_16, (192, ), (1, ))
    assert_size_stride(primals_17, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_18, (384, ), (1, ))
    assert_size_stride(primals_19, (384, ), (1, ))
    assert_size_stride(primals_20, (384, ), (1, ))
    assert_size_stride(primals_21, (384, ), (1, ))
    assert_size_stride(primals_22, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (384, ), (1, ))
    assert_size_stride(primals_24, (384, ), (1, ))
    assert_size_stride(primals_25, (384, ), (1, ))
    assert_size_stride(primals_26, (384, ), (1, ))
    assert_size_stride(primals_27, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_33, (384, ), (1, ))
    assert_size_stride(primals_34, (384, ), (1, ))
    assert_size_stride(primals_35, (384, ), (1, ))
    assert_size_stride(primals_36, (384, ), (1, ))
    assert_size_stride(primals_37, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_38, (384, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_44, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_48, (384, ), (1, ))
    assert_size_stride(primals_49, (384, ), (1, ))
    assert_size_stride(primals_50, (384, ), (1, ))
    assert_size_stride(primals_51, (384, ), (1, ))
    assert_size_stride(primals_52, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_53, (384, ), (1, ))
    assert_size_stride(primals_54, (384, ), (1, ))
    assert_size_stride(primals_55, (384, ), (1, ))
    assert_size_stride(primals_56, (384, ), (1, ))
    assert_size_stride(primals_57, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_58, (192, ), (1, ))
    assert_size_stride(primals_59, (192, ), (1, ))
    assert_size_stride(primals_60, (192, ), (1, ))
    assert_size_stride(primals_61, (192, ), (1, ))
    assert_size_stride(primals_62, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, ), (1, ))
    assert_size_stride(primals_65, (384, ), (1, ))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (384, ), (1, ))
    assert_size_stride(primals_69, (384, ), (1, ))
    assert_size_stride(primals_70, (384, ), (1, ))
    assert_size_stride(primals_71, (384, ), (1, ))
    assert_size_stride(primals_72, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_73, (192, ), (1, ))
    assert_size_stride(primals_74, (192, ), (1, ))
    assert_size_stride(primals_75, (192, ), (1, ))
    assert_size_stride(primals_76, (192, ), (1, ))
    assert_size_stride(primals_77, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (384, ), (1, ))
    assert_size_stride(primals_80, (384, ), (1, ))
    assert_size_stride(primals_81, (384, ), (1, ))
    assert_size_stride(primals_82, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_83, (384, ), (1, ))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, ), (1, ))
    assert_size_stride(primals_86, (384, ), (1, ))
    assert_size_stride(primals_87, (192, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_88, (192, ), (1, ))
    assert_size_stride(primals_89, (192, ), (1, ))
    assert_size_stride(primals_90, (192, ), (1, ))
    assert_size_stride(primals_91, (192, ), (1, ))
    assert_size_stride(primals_92, (384, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_93, (384, ), (1, ))
    assert_size_stride(primals_94, (384, ), (1, ))
    assert_size_stride(primals_95, (384, ), (1, ))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_98, (384, ), (1, ))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, ), (1, ))
    assert_size_stride(primals_101, (384, ), (1, ))
    assert_size_stride(primals_102, (384, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_103, (384, ), (1, ))
    assert_size_stride(primals_104, (384, ), (1, ))
    assert_size_stride(primals_105, (384, ), (1, ))
    assert_size_stride(primals_106, (384, ), (1, ))
    assert_size_stride(primals_107, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_108, (1536, ), (1, ))
    assert_size_stride(primals_109, (1536, ), (1, ))
    assert_size_stride(primals_110, (1536, ), (1, ))
    assert_size_stride(primals_111, (1536, ), (1, ))
    assert_size_stride(primals_112, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (1536, ), (1, ))
    assert_size_stride(primals_114, (1536, ), (1, ))
    assert_size_stride(primals_115, (1536, ), (1, ))
    assert_size_stride(primals_116, (1536, ), (1, ))
    assert_size_stride(primals_117, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_118, (384, ), (1, ))
    assert_size_stride(primals_119, (384, ), (1, ))
    assert_size_stride(primals_120, (384, ), (1, ))
    assert_size_stride(primals_121, (384, ), (1, ))
    assert_size_stride(primals_122, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_123, (1536, ), (1, ))
    assert_size_stride(primals_124, (1536, ), (1, ))
    assert_size_stride(primals_125, (1536, ), (1, ))
    assert_size_stride(primals_126, (1536, ), (1, ))
    assert_size_stride(primals_127, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (1536, ), (1, ))
    assert_size_stride(primals_129, (1536, ), (1, ))
    assert_size_stride(primals_130, (1536, ), (1, ))
    assert_size_stride(primals_131, (1536, ), (1, ))
    assert_size_stride(primals_132, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_133, (384, ), (1, ))
    assert_size_stride(primals_134, (384, ), (1, ))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, ), (1, ))
    assert_size_stride(primals_137, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_138, (1536, ), (1, ))
    assert_size_stride(primals_139, (1536, ), (1, ))
    assert_size_stride(primals_140, (1536, ), (1, ))
    assert_size_stride(primals_141, (1536, ), (1, ))
    assert_size_stride(primals_142, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_143, (1536, ), (1, ))
    assert_size_stride(primals_144, (1536, ), (1, ))
    assert_size_stride(primals_145, (1536, ), (1, ))
    assert_size_stride(primals_146, (1536, ), (1, ))
    assert_size_stride(primals_147, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_148, (384, ), (1, ))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_153, (1536, ), (1, ))
    assert_size_stride(primals_154, (1536, ), (1, ))
    assert_size_stride(primals_155, (1536, ), (1, ))
    assert_size_stride(primals_156, (1536, ), (1, ))
    assert_size_stride(primals_157, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (1536, ), (1, ))
    assert_size_stride(primals_159, (1536, ), (1, ))
    assert_size_stride(primals_160, (1536, ), (1, ))
    assert_size_stride(primals_161, (1536, ), (1, ))
    assert_size_stride(primals_162, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (384, ), (1, ))
    assert_size_stride(primals_167, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_168, (1536, ), (1, ))
    assert_size_stride(primals_169, (1536, ), (1, ))
    assert_size_stride(primals_170, (1536, ), (1, ))
    assert_size_stride(primals_171, (1536, ), (1, ))
    assert_size_stride(primals_172, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_173, (1536, ), (1, ))
    assert_size_stride(primals_174, (1536, ), (1, ))
    assert_size_stride(primals_175, (1536, ), (1, ))
    assert_size_stride(primals_176, (1536, ), (1, ))
    assert_size_stride(primals_177, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_178, (384, ), (1, ))
    assert_size_stride(primals_179, (384, ), (1, ))
    assert_size_stride(primals_180, (384, ), (1, ))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_182, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_183, (1536, ), (1, ))
    assert_size_stride(primals_184, (1536, ), (1, ))
    assert_size_stride(primals_185, (1536, ), (1, ))
    assert_size_stride(primals_186, (1536, ), (1, ))
    assert_size_stride(primals_187, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_188, (1536, ), (1, ))
    assert_size_stride(primals_189, (1536, ), (1, ))
    assert_size_stride(primals_190, (1536, ), (1, ))
    assert_size_stride(primals_191, (1536, ), (1, ))
    assert_size_stride(primals_192, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (384, ), (1, ))
    assert_size_stride(primals_196, (384, ), (1, ))
    assert_size_stride(primals_197, (768, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_198, (768, ), (1, ))
    assert_size_stride(primals_199, (768, ), (1, ))
    assert_size_stride(primals_200, (768, ), (1, ))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (768, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_203, (768, ), (1, ))
    assert_size_stride(primals_204, (768, ), (1, ))
    assert_size_stride(primals_205, (768, ), (1, ))
    assert_size_stride(primals_206, (768, ), (1, ))
    assert_size_stride(primals_207, (64, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_208, (64, ), (1, ))
    assert_size_stride(primals_209, (64, ), (1, ))
    assert_size_stride(primals_210, (64, ), (1, ))
    assert_size_stride(primals_211, (64, ), (1, ))
    assert_size_stride(primals_212, (128, 67, 3, 3), (603, 9, 3, 1))
    assert_size_stride(primals_213, (128, ), (1, ))
    assert_size_stride(primals_214, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_215, (128, ), (1, ))
    assert_size_stride(primals_216, (128, ), (1, ))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (256, 131, 3, 3), (1179, 9, 3, 1))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (512, 256, 7, 7), (12544, 49, 7, 1))
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (512, ), (1, ))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (8, 19328), (19328, 1))
    assert_size_stride(primals_227, (8, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 4, 4, 4), (64, 1, 16, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 16, 16, grid=grid(16, 16), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((16, 4, 3, 3), (36, 1, 12, 4), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 64, 9, grid=grid(64, 9), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((128, 67, 3, 3), (603, 1, 201, 67), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_212, buf2, 8576, 9, grid=grid(8576, 9), stream=stream0)
        del primals_212
        buf3 = empty_strided_cuda((128, 128, 3, 3), (1152, 1, 384, 128), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_214, buf3, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_214
        buf4 = empty_strided_cuda((256, 131, 3, 3), (1179, 1, 393, 131), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_219, buf4, 33536, 9, grid=grid(33536, 9), stream=stream0)
        del primals_219
        buf5 = empty_strided_cuda((512, 256, 7, 7), (12544, 1, 1792, 256), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_5.run(primals_221, buf5, 131072, 49, grid=grid(131072, 49), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 16, 2, 2), (64, 1, 32, 16))
        buf7 = empty_strided_cuda((4, 16, 2, 2), (64, 1, 32, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_2, input_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6.run(buf6, primals_3, primals_4, primals_5, primals_6, buf7, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_4], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf8, (4, 16, 2, 2), (64, 1, 32, 16))
        buf9 = empty_strided_cuda((4, 16, 2, 2), (64, 1, 32, 16), torch.float32)
        # Topologically Sorted Source Nodes: [input_5, input_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_6.run(buf8, primals_8, primals_9, primals_10, primals_11, buf9, 256, grid=grid(256), stream=stream0)
        # Topologically Sorted Source Nodes: [input_7], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 192, 2, 2), (768, 1, 384, 192))
        buf11 = empty_strided_cuda((4, 192, 2, 2), (768, 1, 384, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf10, primals_13, primals_14, primals_15, primals_16, buf11, 3072, grid=grid(3072), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [input_9], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 384, 2, 2), (1536, 1, 768, 384))
        buf13 = empty_strided_cuda((4, 384, 2, 2), (1536, 1, 768, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_10, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_8.run(buf12, primals_18, primals_19, primals_20, primals_21, buf13, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_12], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_22, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf14, (4, 384, 1, 1), (384, 1, 384, 384))
        buf15 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_13, input_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf14, primals_23, primals_24, primals_25, primals_26, buf15, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_15], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 192, 1, 1), (192, 1, 192, 192))
        buf17 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_16], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf16, primals_28, primals_29, primals_30, primals_31, buf17, 768, grid=grid(768), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [input_17], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_32, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 384, 1, 1), (384, 1, 384, 384))
        buf19 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf18, primals_33, primals_34, primals_35, primals_36, buf19, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_20], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_37, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf20, (4, 384, 1, 1), (384, 1, 384, 384))
        buf21 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_21, input_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf20, primals_38, primals_39, primals_40, primals_41, buf21, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_23], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 192, 1, 1), (192, 1, 192, 192))
        buf23 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf17, buf22, primals_43, primals_44, primals_45, primals_46, buf23, 768, grid=grid(768), stream=stream0)
        del primals_46
        # Topologically Sorted Source Nodes: [input_26], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 384, 1, 1), (384, 1, 384, 384))
        buf25 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_27, input_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf24, primals_48, primals_49, primals_50, primals_51, buf25, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_29], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf26, (4, 384, 1, 1), (384, 1, 384, 384))
        buf27 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf26, primals_53, primals_54, primals_55, primals_56, buf27, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_32], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 192, 1, 1), (192, 1, 192, 192))
        buf29 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_33, input_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf23, buf28, primals_58, primals_59, primals_60, primals_61, buf29, 768, grid=grid(768), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [input_35], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 384, 1, 1), (384, 1, 384, 384))
        buf31 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf30, primals_63, primals_64, primals_65, primals_66, buf31, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_38], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf32, (4, 384, 1, 1), (384, 1, 384, 384))
        buf33 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_39, input_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf32, primals_68, primals_69, primals_70, primals_71, buf33, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_41], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 192, 1, 1), (192, 1, 192, 192))
        buf35 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf29, buf34, primals_73, primals_74, primals_75, primals_76, buf35, 768, grid=grid(768), stream=stream0)
        del primals_76
        # Topologically Sorted Source Nodes: [input_44], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 384, 1, 1), (384, 1, 384, 384))
        buf37 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_45, input_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf36, primals_78, primals_79, primals_80, primals_81, buf37, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_47], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf38, (4, 384, 1, 1), (384, 1, 384, 384))
        buf39 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf38, primals_83, primals_84, primals_85, primals_86, buf39, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_50], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 192, 1, 1), (192, 1, 192, 192))
        buf41 = empty_strided_cuda((4, 192, 1, 1), (192, 1, 192, 192), torch.float32)
        # Topologically Sorted Source Nodes: [input_51, input_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_11.run(buf35, buf40, primals_88, primals_89, primals_90, primals_91, buf41, 768, grid=grid(768), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [input_53], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_92, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 384, 1, 1), (384, 1, 384, 384))
        buf43 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf42, primals_93, primals_94, primals_95, primals_96, buf43, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_56], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_97, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf44, (4, 384, 1, 1), (384, 1, 384, 384))
        buf45 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_57, input_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_9.run(buf44, primals_98, primals_99, primals_100, primals_101, buf45, 1536, grid=grid(1536), stream=stream0)
        # Topologically Sorted Source Nodes: [input_59], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_102, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 384, 1, 1), (384, 1, 384, 384))
        buf47 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_60], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf46, primals_103, primals_104, primals_105, primals_106, buf47, 1536, grid=grid(1536), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [input_61], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf49 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf48, primals_108, primals_109, primals_110, primals_111, buf49, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_64], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf50, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf51 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_65, input_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf50, primals_113, primals_114, primals_115, primals_116, buf51, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_67], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 384, 1, 1), (384, 1, 384, 384))
        buf53 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf47, buf52, primals_118, primals_119, primals_120, primals_121, buf53, 1536, grid=grid(1536), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [input_70], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf55 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_71, input_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf54, primals_123, primals_124, primals_125, primals_126, buf55, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_73], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf56, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf57 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_74, input_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf56, primals_128, primals_129, primals_130, primals_131, buf57, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_76], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 384, 1, 1), (384, 1, 384, 384))
        buf59 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_77, input_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf53, buf58, primals_133, primals_134, primals_135, primals_136, buf59, 1536, grid=grid(1536), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [input_79], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_137, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf61 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf60, primals_138, primals_139, primals_140, primals_141, buf61, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_82], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf62, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf63 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_83, input_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf62, primals_143, primals_144, primals_145, primals_146, buf63, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_85], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_147, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 384, 1, 1), (384, 1, 384, 384))
        buf65 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_86, input_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf59, buf64, primals_148, primals_149, primals_150, primals_151, buf65, 1536, grid=grid(1536), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [input_88], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf67 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_89, input_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf66, primals_153, primals_154, primals_155, primals_156, buf67, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_91], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf68, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf69 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf68, primals_158, primals_159, primals_160, primals_161, buf69, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_94], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 384, 1, 1), (384, 1, 384, 384))
        buf71 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_95, input_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf65, buf70, primals_163, primals_164, primals_165, primals_166, buf71, 1536, grid=grid(1536), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [input_97], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_167, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf73 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_98, input_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf72, primals_168, primals_169, primals_170, primals_171, buf73, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_100], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf74, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf75 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_101, input_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf74, primals_173, primals_174, primals_175, primals_176, buf75, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_103], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_177, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 384, 1, 1), (384, 1, 384, 384))
        buf77 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf71, buf76, primals_178, primals_179, primals_180, primals_181, buf77, 1536, grid=grid(1536), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [input_106], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf79 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_107, input_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf78, primals_183, primals_184, primals_185, primals_186, buf79, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_109], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf80, (4, 1536, 1, 1), (1536, 1, 1536, 1536))
        buf81 = empty_strided_cuda((4, 1536, 1, 1), (1536, 1, 1536, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_13.run(buf80, primals_188, primals_189, primals_190, primals_191, buf81, 6144, grid=grid(6144), stream=stream0)
        # Topologically Sorted Source Nodes: [input_112], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 384, 1, 1), (384, 1, 384, 384))
        buf83 = empty_strided_cuda((4, 384, 1, 1), (384, 1, 384, 384), torch.float32)
        # Topologically Sorted Source Nodes: [input_113, input_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_14.run(buf77, buf82, primals_193, primals_194, primals_195, primals_196, buf83, 1536, grid=grid(1536), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [input_115], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_197, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 768, 1, 1), (768, 1, 768, 768))
        buf85 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 768, 768), torch.float32)
        # Topologically Sorted Source Nodes: [input_116, input_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf84, primals_198, primals_199, primals_200, primals_201, buf85, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_118], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=768, bias=None)
        assert_size_stride(buf86, (4, 768, 1, 1), (768, 1, 768, 768))
        buf87 = empty_strided_cuda((4, 768, 1, 1), (768, 1, 768, 768), torch.float32)
        # Topologically Sorted Source Nodes: [input_119, input_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_15.run(buf86, primals_203, primals_204, primals_205, primals_206, buf87, 3072, grid=grid(3072), stream=stream0)
        # Topologically Sorted Source Nodes: [input_121], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_207, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 64, 1, 1), (64, 1, 64, 64))
        buf89 = empty_strided_cuda((4, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        # Topologically Sorted Source Nodes: [input_122], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf88, primals_208, primals_209, primals_210, primals_211, buf89, 256, grid=grid(256), stream=stream0)
        del primals_211
        buf90 = empty_strided_cuda((4, 67, 14, 14), (13132, 1, 938, 67), torch.float32)
        # Topologically Sorted Source Nodes: [ret_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_17.run(buf89, buf90, 52528, grid=grid(52528), stream=stream0)
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 128, 14, 14), (25088, 1, 1792, 128))
        buf92 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [ret_2], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_18.run(buf92, primals_213, 100352, grid=grid(100352), stream=stream0)
        del primals_213
        # Topologically Sorted Source Nodes: [input_124], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, buf3, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 128, 7, 7), (6272, 1, 896, 128))
        buf94 = empty_strided_cuda((4, 128, 7, 7), (6272, 49, 7, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_125, input_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardtanh]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_hardtanh_19.run(buf93, primals_215, primals_216, primals_217, primals_218, buf94, 512, 49, grid=grid(512, 49), stream=stream0)
        buf95 = empty_strided_cuda((4, 131, 7, 7), (6419, 1, 917, 131), torch.float32)
        # Topologically Sorted Source Nodes: [ret_4], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_20.run(buf94, buf95, 25676, grid=grid(25676), stream=stream0)
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 256, 7, 7), (12544, 1, 1792, 256))
        buf97 = buf96; del buf96  # reuse
        # Topologically Sorted Source Nodes: [ret_5], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_21.run(buf97, primals_220, 50176, grid=grid(50176), stream=stream0)
        del primals_220
        # Topologically Sorted Source Nodes: [input_127], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, buf5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 512, 1, 1), (512, 1, 512, 512))
        buf99 = empty_strided_cuda((4, 19328), (19328, 1), torch.float32)
        # Topologically Sorted Source Nodes: [tensors], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_22.run(buf89, buf94, buf98, primals_222, primals_223, primals_224, primals_225, buf99, 77312, grid=grid(77312), stream=stream0)
        del buf94
        buf100 = empty_strided_cuda((4, 8), (8, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_130], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_227, buf99, reinterpret_tensor(primals_226, (19328, 8), (1, 19328), 0), alpha=1, beta=1, out=buf100)
        del primals_227
    return (reinterpret_tensor(buf100, (4, 4, 2), (8, 2, 1), 0), buf0, buf1, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, buf2, buf3, primals_215, primals_216, primals_217, primals_218, buf4, buf5, primals_222, primals_223, primals_224, primals_225, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf92, buf93, buf95, buf97, buf98, buf99, primals_226, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 4, 4, 4), (64, 16, 4, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((192, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((192, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((384, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((768, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((64, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, 67, 3, 3), (603, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, 131, 3, 3), (1179, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((512, 256, 7, 7), (12544, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((8, 19328), (19328, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
