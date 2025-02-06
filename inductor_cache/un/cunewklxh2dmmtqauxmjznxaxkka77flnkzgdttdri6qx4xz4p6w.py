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


# kernel path: inductor_cache/fr/cfrqz7giy7xpgq6ebpine5k355oxpl2bcs25dikp5emaek2jc2cb.py
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
    size_hints={'y': 4096, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
    xnumel = 4096
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 1024)
    y1 = yindex // 1024
    tmp0 = tl.load(in_ptr0 + (x2 + 4096*y3), None)
    tl.store(out_ptr0 + (y0 + 1024*x2 + 4194304*y1), tmp0, None)
''', device_str='cuda')


# kernel path: inductor_cache/md/cmdrrusw74ecs75x6xq7egvacvox3r3g7nf5c5xt5mrqi6gcokvo.py
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
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 43008
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1344*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/dx/cdxufc3ktefob4cagwokao5khdq7ljgv2azdstcxczttw7msco63.py
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
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 57344
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 224)
    y1 = yindex // 224
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 224*x2 + 1568*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/34/c34ypzny3sm5kwdsr4vftsdsbhpiekvie64fqmhyxcnodi2vynk3.py
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
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 192)
    y1 = yindex // 192
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 192*x2 + 1344*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/i3/ci3zxqlyigivyk7oyrk7tksp2lanlqu3dosfolxtwzye6df7yysv.py
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
    size_hints={'y': 65536, 'x': 8}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 7
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 224)
    y1 = yindex // 224
    tmp0 = tl.load(in_ptr0 + (x2 + 7*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 224*x2 + 1568*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/ws/cwshe44szyokovya63t6njmes7f2nuvtcchftnsszbixdw4f235o.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_4 => add_3, mul_4, mul_5, sub_1
#   x_5 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 192)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/ji/cjiy6ig6csvz4jxvue4vclhrl6v5cgrpavkodfapb53v6tnvzm2m.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_7 => add_5, mul_7, mul_8, sub_2
#   x_8 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3670016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 224)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: inductor_cache/p7/cp7szdbsnlr5575dori7ozkt77wmqzgmc2iv6uvgnuhu5juptb2k.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
# Source node to ATen node mapping:
#   input_1 => avg_pool2d
# Graph fragment:
#   %avg_pool2d : [num_users=2] = call_function[target=torch.ops.aten.avg_pool2d.default](args = (%primals_2, [3, 3], [1, 1], [1, 1], False, False), kwargs = {})
triton_poi_fused_avg_pool2d_7 = async_compile.triton('triton_poi_fused_avg_pool2d_7', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_avg_pool2d_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 65536) % 64)
    x1 = ((xindex // 1024) % 64)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-66560) + x6), tmp10, other=0.0)
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-65536) + x6), tmp16, other=0.0)
    tmp18 = tmp17 + tmp11
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-64512) + x6), tmp23, other=0.0)
    tmp25 = tmp24 + tmp18
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1024) + x6), tmp30, other=0.0)
    tmp32 = tmp31 + tmp25
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), tmp33, other=0.0)
    tmp35 = tmp34 + tmp32
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1024 + x6), tmp36, other=0.0)
    tmp38 = tmp37 + tmp35
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (64512 + x6), tmp43, other=0.0)
    tmp45 = tmp44 + tmp38
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (65536 + x6), tmp46, other=0.0)
    tmp48 = tmp47 + tmp45
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (66560 + x6), tmp49, other=0.0)
    tmp51 = tmp50 + tmp48
    tmp52 = ((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0))) + ((64) * ((64) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (64)))*((64) * ((64) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (64))) + ((-1)*((0) * ((0) >= ((-1) + x1)) + ((-1) + x1) * (((-1) + x1) > (0)))*((64) * ((64) <= (2 + x2)) + (2 + x2) * ((2 + x2) < (64)))) + ((-1)*((0) * ((0) >= ((-1) + x2)) + ((-1) + x2) * (((-1) + x2) > (0)))*((64) * ((64) <= (2 + x1)) + (2 + x1) * ((2 + x1) < (64))))
    tmp53 = tmp51 / tmp52
    tl.store(out_ptr0 + (x6), tmp53, None)
''', device_str='cuda')


# kernel path: inductor_cache/f4/cf4pey42jhnhxse6s4s7b7o6owuo42zrfxfxtb5wwsxm4amoxcv6.py
# Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   out => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%relu, %relu_3, %relu_8, %relu_9], 1), kwargs = {})
triton_poi_fused_cat_8 = async_compile.triton('triton_poi_fused_cat_8', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'in_ptr10': '*fp32', 'in_ptr11': '*fp32', 'in_ptr12': '*fp32', 'in_ptr13': '*fp32', 'in_ptr14': '*fp32', 'in_ptr15': '*fp32', 'in_ptr16': '*fp32', 'in_ptr17': '*fp32', 'in_ptr18': '*fp32', 'in_ptr19': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 20, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 4096) % 1024)
    x0 = (xindex % 4096)
    x2 = xindex // 4194304
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (384*x0 + 1572864*x2 + (x1)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = tl.full([1], 1, tl.int32)
    tmp13 = tmp12 / tmp11
    tmp14 = 1.0
    tmp15 = tmp13 * tmp14
    tmp16 = tmp7 * tmp15
    tmp17 = tl.load(in_ptr3 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp18 = tmp16 * tmp17
    tmp19 = tl.load(in_ptr4 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp20)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = tl.full([1], 640, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = tmp25 & tmp27
    tmp29 = tl.load(in_ptr5 + (256*x0 + 1048576*x2 + ((-384) + x1)), tmp28, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr6 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 - tmp30
    tmp32 = tl.load(in_ptr7 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp33 = 0.001
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.sqrt(tmp34)
    tmp36 = tl.full([1], 1, tl.int32)
    tmp37 = tmp36 / tmp35
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp31 * tmp39
    tmp41 = tl.load(in_ptr8 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp42 = tmp40 * tmp41
    tmp43 = tl.load(in_ptr9 + ((-384) + x1), tmp28, eviction_policy='evict_last', other=0.0)
    tmp44 = tmp42 + tmp43
    tmp45 = tl.full([1], 0, tl.int32)
    tmp46 = triton_helpers.maximum(tmp45, tmp44)
    tmp47 = tl.full(tmp46.shape, 0.0, tmp46.dtype)
    tmp48 = tl.where(tmp28, tmp46, tmp47)
    tmp49 = tmp0 >= tmp26
    tmp50 = tl.full([1], 896, tl.int64)
    tmp51 = tmp0 < tmp50
    tmp52 = tmp49 & tmp51
    tmp53 = tl.load(in_ptr10 + (256*x0 + 1048576*x2 + ((-640) + x1)), tmp52, eviction_policy='evict_last', other=0.0)
    tmp54 = tl.load(in_ptr11 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp55 = tmp53 - tmp54
    tmp56 = tl.load(in_ptr12 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp57 = 0.001
    tmp58 = tmp56 + tmp57
    tmp59 = libdevice.sqrt(tmp58)
    tmp60 = tl.full([1], 1, tl.int32)
    tmp61 = tmp60 / tmp59
    tmp62 = 1.0
    tmp63 = tmp61 * tmp62
    tmp64 = tmp55 * tmp63
    tmp65 = tl.load(in_ptr13 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp66 = tmp64 * tmp65
    tmp67 = tl.load(in_ptr14 + ((-640) + x1), tmp52, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 + tmp67
    tmp69 = tl.full([1], 0, tl.int32)
    tmp70 = triton_helpers.maximum(tmp69, tmp68)
    tmp71 = tl.full(tmp70.shape, 0.0, tmp70.dtype)
    tmp72 = tl.where(tmp52, tmp70, tmp71)
    tmp73 = tmp0 >= tmp50
    tmp74 = tl.full([1], 1024, tl.int64)
    tmp75 = tmp0 < tmp74
    tmp76 = tl.load(in_ptr15 + (128*x0 + 524288*x2 + ((-896) + x1)), tmp73, eviction_policy='evict_last', other=0.0)
    tmp77 = tl.load(in_ptr16 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 - tmp77
    tmp79 = tl.load(in_ptr17 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp80 = 0.001
    tmp81 = tmp79 + tmp80
    tmp82 = libdevice.sqrt(tmp81)
    tmp83 = tl.full([1], 1, tl.int32)
    tmp84 = tmp83 / tmp82
    tmp85 = 1.0
    tmp86 = tmp84 * tmp85
    tmp87 = tmp78 * tmp86
    tmp88 = tl.load(in_ptr18 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp89 = tmp87 * tmp88
    tmp90 = tl.load(in_ptr19 + ((-896) + x1), tmp73, eviction_policy='evict_last', other=0.0)
    tmp91 = tmp89 + tmp90
    tmp92 = tl.full([1], 0, tl.int32)
    tmp93 = triton_helpers.maximum(tmp92, tmp91)
    tmp94 = tl.full(tmp93.shape, 0.0, tmp93.dtype)
    tmp95 = tl.where(tmp73, tmp93, tmp94)
    tmp96 = tl.where(tmp52, tmp72, tmp95)
    tmp97 = tl.where(tmp28, tmp48, tmp96)
    tmp98 = tl.where(tmp4, tmp24, tmp97)
    tl.store(out_ptr0 + (x3), tmp98, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51 = args
    args.clear()
    assert_size_stride(primals_1, (384, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_2, (4, 1024, 64, 64), (4194304, 4096, 64, 1))
    assert_size_stride(primals_3, (384, ), (1, ))
    assert_size_stride(primals_4, (384, ), (1, ))
    assert_size_stride(primals_5, (384, ), (1, ))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_8, (192, ), (1, ))
    assert_size_stride(primals_9, (192, ), (1, ))
    assert_size_stride(primals_10, (192, ), (1, ))
    assert_size_stride(primals_11, (192, ), (1, ))
    assert_size_stride(primals_12, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_13, (224, ), (1, ))
    assert_size_stride(primals_14, (224, ), (1, ))
    assert_size_stride(primals_15, (224, ), (1, ))
    assert_size_stride(primals_16, (224, ), (1, ))
    assert_size_stride(primals_17, (256, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, ), (1, ))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (192, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_23, (192, ), (1, ))
    assert_size_stride(primals_24, (192, ), (1, ))
    assert_size_stride(primals_25, (192, ), (1, ))
    assert_size_stride(primals_26, (192, ), (1, ))
    assert_size_stride(primals_27, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (224, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(primals_33, (224, ), (1, ))
    assert_size_stride(primals_34, (224, ), (1, ))
    assert_size_stride(primals_35, (224, ), (1, ))
    assert_size_stride(primals_36, (224, ), (1, ))
    assert_size_stride(primals_37, (224, 224, 7, 1), (1568, 7, 1, 1))
    assert_size_stride(primals_38, (224, ), (1, ))
    assert_size_stride(primals_39, (224, ), (1, ))
    assert_size_stride(primals_40, (224, ), (1, ))
    assert_size_stride(primals_41, (224, ), (1, ))
    assert_size_stride(primals_42, (256, 224, 1, 7), (1568, 7, 7, 1))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (128, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((4, 1024, 64, 64), (4194304, 1, 65536, 1024), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_2, buf0, 4096, 4096, grid=grid(4096, 4096), stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((224, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_12, buf1, 43008, 7, grid=grid(43008, 7), stream=stream0)
        del primals_12
        buf2 = empty_strided_cuda((256, 224, 7, 1), (1568, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_17, buf2, 57344, 7, grid=grid(57344, 7), stream=stream0)
        del primals_17
        buf3 = empty_strided_cuda((192, 192, 7, 1), (1344, 1, 192, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_3.run(primals_27, buf3, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del primals_27
        buf4 = empty_strided_cuda((224, 192, 1, 7), (1344, 1, 1344, 192), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_32, buf4, 43008, 7, grid=grid(43008, 7), stream=stream0)
        del primals_32
        buf5 = empty_strided_cuda((224, 224, 7, 1), (1568, 1, 224, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_4.run(primals_37, buf5, 50176, 7, grid=grid(50176, 7), stream=stream0)
        del primals_37
        buf6 = empty_strided_cuda((256, 224, 1, 7), (1568, 1, 1568, 224), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_42, buf6, 57344, 7, grid=grid(57344, 7), stream=stream0)
        del primals_42
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf0, primals_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 384, 64, 64), (1572864, 1, 24576, 384))
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf0, primals_7, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf9 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf8, primals_8, primals_9, primals_10, primals_11, buf9, 3145728, grid=grid(3145728), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, buf1, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 224, 64, 64), (917504, 1, 14336, 224))
        buf11 = empty_strided_cuda((4, 224, 64, 64), (917504, 1, 14336, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf10, primals_13, primals_14, primals_15, primals_16, buf11, 3670016, grid=grid(3670016), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, buf2, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 256, 64, 64), (1048576, 1, 16384, 256))
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf0, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf14 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_13, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf13, primals_23, primals_24, primals_25, primals_26, buf14, 3145728, grid=grid(3145728), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [x_15], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, buf3, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 192, 64, 64), (786432, 1, 12288, 192))
        buf16 = empty_strided_cuda((4, 192, 64, 64), (786432, 1, 12288, 192), torch.float32)
        # Topologically Sorted Source Nodes: [x_16, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf15, primals_28, primals_29, primals_30, primals_31, buf16, 3145728, grid=grid(3145728), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, buf4, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 224, 64, 64), (917504, 1, 14336, 224))
        buf18 = empty_strided_cuda((4, 224, 64, 64), (917504, 1, 14336, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf17, primals_33, primals_34, primals_35, primals_36, buf18, 3670016, grid=grid(3670016), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, buf5, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 224, 64, 64), (917504, 1, 14336, 224))
        buf20 = empty_strided_cuda((4, 224, 64, 64), (917504, 1, 14336, 224), torch.float32)
        # Topologically Sorted Source Nodes: [x_22, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf19, primals_38, primals_39, primals_40, primals_41, buf20, 3670016, grid=grid(3670016), stream=stream0)
        del primals_41
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, buf6, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 256, 64, 64), (1048576, 1, 16384, 256))
        buf22 = empty_strided_cuda((4, 1024, 64, 64), (4194304, 1, 65536, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten.avg_pool2d]
        stream0 = get_raw_stream(0)
        triton_poi_fused_avg_pool2d_7.run(buf0, buf22, 16777216, grid=grid(16777216), stream=stream0)
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 128, 64, 64), (524288, 1, 8192, 128))
        buf24 = empty_strided_cuda((4, 1024, 64, 64), (4194304, 4096, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [out], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_8.run(buf7, primals_3, primals_4, primals_5, primals_6, buf12, primals_18, primals_19, primals_20, primals_21, buf21, primals_43, primals_44, primals_45, primals_46, buf23, primals_48, primals_49, primals_50, primals_51, buf24, 16777216, grid=grid(16777216), stream=stream0)
    return (buf24, primals_1, buf0, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, buf1, primals_13, primals_14, primals_15, buf2, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, buf3, primals_28, primals_29, primals_30, buf4, primals_33, primals_34, primals_35, buf5, primals_38, primals_39, primals_40, buf6, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((384, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 1024, 64, 64), (4194304, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((192, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((224, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((224, 224, 7, 1), (1568, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, 224, 1, 7), (1568, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
