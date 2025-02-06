# AOT ID: ['59_forward']
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


# kernel path: inductor_cache/ks/cksa7rovu3ro7bt4vrd6ghsjths23wrsv7asy7t26bbd7ihalfqw.py
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
    size_hints={'y': 128, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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


# kernel path: inductor_cache/f3/cf3yvvrx2dp4pn5dwcyzm6qhg7y76yekqxcww4ry23bgzk3jew7k.py
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
    size_hints={'y': 16, 'x': 4096}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 3), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (y0 + 3*x2 + 12288*y1), tmp0, ymask)
''', device_str='cuda')


# kernel path: inductor_cache/h6/ch6idif7qwul5cpmntwlb6pzuk7uvdh5cx5qbyj7c3k72t6saeli.py
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
    size_hints={'y': 2048, 'x': 16}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 32)
    y1 = yindex // 32
    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 32*x2 + 288*y1), tmp0, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/j2/cj22l6baypvfioaksifra76camsgvvwpnltw3l7sdxvthabqlmwh.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_1 => add_1, mul_1, mul_2, sub
#   x_2 => relu
# Graph fragment:
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution, %unsqueeze_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %unsqueeze_3), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %unsqueeze_5), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %unsqueeze_7), kwargs = {})
#   %relu : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_1,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_3', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 123008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/26/c265amiqica2eue47jvxc7mrmj7pcdbrpjinvldvhmptk5xffwh4.py
# Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_4 => add_3, mul_4, mul_5, sub_1
#   x_5 => relu_1
# Graph fragment:
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_1, %unsqueeze_9), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %unsqueeze_11), kwargs = {})
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_4, %unsqueeze_13), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_5, %unsqueeze_15), kwargs = {})
#   %relu_1 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_3,), kwargs = {})
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
    xnumel = 215296
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bs/cbs4vug476v4gw4qmtgxkahdzgdn7eeob4wcesc65c6f3wzy6qrq.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_5, mul_7, mul_8, sub_2
#   input_2 => relu_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_5,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_5', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 430592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zs/czs4rm3cgiu37cfg63mlrf22j2jy3blvufb25qlzbmvgeufcuihy.py
# Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_3 => add_7, mul_10, mul_11, sub_3
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_5, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_6', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 430592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 128)
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


# kernel path: inductor_cache/5a/c5a6p5oza4hgfdaeirgoykflot3jmqo7alaadtd2kps5stzzsqjy.py
# Topologically Sorted Source Nodes: [input_4, skip_1, x_10, input_5], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_4 => _low_memory_max_pool2d_with_offsets, getitem_1
#   input_5 => relu_3
#   skip_1 => add_9, mul_13, mul_14, sub_4
#   x_10 => add_10
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_7, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets, 1), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
#   %add_10 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, %add_9), kwargs = {})
#   %relu_3 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_10,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_7', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 115200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 1920) % 15)
    x1 = ((xindex // 128) % 15)
    x0 = (xindex % 128)
    x3 = xindex // 28800
    x5 = xindex
    tmp77 = tl.load(in_ptr1 + (x5), xmask)
    tmp78 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 29, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-3840) + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-3712) + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3584) + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-128) + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 256*x1 + 7424*x2 + 107648*x3), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (128 + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3584 + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (3712 + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (3840 + x0 + 256*x1 + 7424*x2 + 107648*x3), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp79 = tmp77 - tmp78
    tmp81 = 1e-05
    tmp82 = tmp80 + tmp81
    tmp83 = libdevice.sqrt(tmp82)
    tmp84 = tl.full([1], 1, tl.int32)
    tmp85 = tmp84 / tmp83
    tmp86 = 1.0
    tmp87 = tmp85 * tmp86
    tmp88 = tmp79 * tmp87
    tmp90 = tmp88 * tmp89
    tmp92 = tmp90 + tmp91
    tmp93 = tmp51 + tmp92
    tmp94 = tl.full([1], 0, tl.int32)
    tmp95 = triton_helpers.maximum(tmp94, tmp93)
    tl.store(out_ptr0 + (x5), tmp76, xmask)
    tl.store(in_out_ptr0 + (x5), tmp93, xmask)
    tl.store(out_ptr1 + (x5), tmp95, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/k5/ck5y7iey2lf6dig3drfkrryobdbwo73kg7p5v4mgyo5pll7av2h7.py
# Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_12, mul_16, mul_17, sub_5
#   input_7 => relu_4
# Graph fragment:
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_41), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %unsqueeze_43), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %unsqueeze_45), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %unsqueeze_47), kwargs = {})
#   %relu_4 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_12,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/f6/cf6h6ujvx5mculcq7skkx33bwm4bgokkugh4c4zfppnnddlvnuwi.py
# Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_8 => add_14, mul_19, mul_20, sub_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_14 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_9', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 230400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 256)
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


# kernel path: inductor_cache/cr/ccrayhhy25522xnzoa5r3d4grrhag2ldw377ofjjtptbpl2sbpqb.py
# Topologically Sorted Source Nodes: [input_9, skip_3, x_15, input_10], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_10 => relu_5
#   input_9 => _low_memory_max_pool2d_with_offsets_1, getitem_3
#   skip_3 => add_16, mul_22, mul_23, sub_7
#   x_15 => add_17
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_1 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_14, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_1, 1), kwargs = {})
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %add_17 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, %add_16), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_17,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_10', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_10', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 2048) % 8)
    x1 = ((xindex // 256) % 8)
    x0 = (xindex % 256)
    x3 = xindex // 16384
    x5 = xindex
    tmp77 = tl.load(in_ptr1 + (x5), None)
    tmp78 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 15, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-4096) + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-3840) + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3584) + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-256) + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 512*x1 + 7680*x2 + 57600*x3), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (256 + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3584 + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (3840 + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (4096 + x0 + 512*x1 + 7680*x2 + 57600*x3), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp79 = tmp77 - tmp78
    tmp81 = 1e-05
    tmp82 = tmp80 + tmp81
    tmp83 = libdevice.sqrt(tmp82)
    tmp84 = tl.full([1], 1, tl.int32)
    tmp85 = tmp84 / tmp83
    tmp86 = 1.0
    tmp87 = tmp85 * tmp86
    tmp88 = tmp79 * tmp87
    tmp90 = tmp88 * tmp89
    tmp92 = tmp90 + tmp91
    tmp93 = tmp51 + tmp92
    tmp94 = tl.full([1], 0, tl.int32)
    tmp95 = triton_helpers.maximum(tmp94, tmp93)
    tl.store(out_ptr0 + (x5), tmp76, None)
    tl.store(in_out_ptr0 + (x5), tmp93, None)
    tl.store(out_ptr1 + (x5), tmp95, None)
''', device_str='cuda')


# kernel path: inductor_cache/o2/co2bepnpgrrtwn544zhsavjnyi5rmb3dygp44np56pvbot6uvup7.py
# Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_11 => add_19, mul_25, mul_26, sub_8
#   input_12 => relu_6
# Graph fragment:
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %relu_6 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_19,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 186368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 728)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/vt/cvtrhrevc6c4kj7fkkxqfkub6up6hpzflyr2fu2vcxoycguwf5vk.py
# Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_13 => add_21, mul_28, mul_29, sub_9
# Graph fragment:
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_73), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %unsqueeze_75), kwargs = {})
#   %mul_29 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_28, %unsqueeze_77), kwargs = {})
#   %add_21 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_29, %unsqueeze_79), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 186368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 728)
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


# kernel path: inductor_cache/ig/cig6dcvobucbyozpxavafnbneswdps5yoyrd3jjcnxsibzspyl4m.py
# Topologically Sorted Source Nodes: [input_14, skip_5, x_20, input_15], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_14 => _low_memory_max_pool2d_with_offsets_2, getitem_5
#   input_15 => relu_7
#   skip_5 => add_23, mul_31, mul_32, sub_10
#   x_20 => add_24
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_2 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_21, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_2, 1), kwargs = {})
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_16, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, %add_23), kwargs = {})
#   %relu_7 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_24,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i8', 'out_ptr1': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 2912) % 4)
    x1 = ((xindex // 728) % 4)
    x0 = (xindex % 728)
    x7 = xindex // 2912
    x5 = xindex
    tmp77 = tl.load(in_ptr1 + (x5), xmask)
    tmp78 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 8, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6552) + x0 + 1456*x1 + 11648*x7), tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-5824) + x0 + 1456*x1 + 11648*x7), tmp16 & xmask, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-5096) + x0 + 1456*x1 + 11648*x7), tmp23 & xmask, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-728) + x0 + 1456*x1 + 11648*x7), tmp30 & xmask, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 1456*x1 + 11648*x7), tmp33 & xmask, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (728 + x0 + 1456*x1 + 11648*x7), tmp36 & xmask, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (5096 + x0 + 1456*x1 + 11648*x7), tmp43 & xmask, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (5824 + x0 + 1456*x1 + 11648*x7), tmp46 & xmask, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (6552 + x0 + 1456*x1 + 11648*x7), tmp49 & xmask, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp79 = tmp77 - tmp78
    tmp81 = 1e-05
    tmp82 = tmp80 + tmp81
    tmp83 = libdevice.sqrt(tmp82)
    tmp84 = tl.full([1], 1, tl.int32)
    tmp85 = tmp84 / tmp83
    tmp86 = 1.0
    tmp87 = tmp85 * tmp86
    tmp88 = tmp79 * tmp87
    tmp90 = tmp88 * tmp89
    tmp92 = tmp90 + tmp91
    tmp93 = tmp51 + tmp92
    tmp94 = tl.full([1], 0, tl.int32)
    tmp95 = triton_helpers.maximum(tmp94, tmp93)
    tl.store(out_ptr0 + (x5), tmp76, xmask)
    tl.store(in_out_ptr0 + (x5), tmp93, xmask)
    tl.store(out_ptr1 + (x5), tmp95, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/db/cdbqvebayjpvlickr4x5xyxoyha34iwtqj7d4za4yboj6qjzymxp.py
# Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_26, mul_34, mul_35, sub_11
#   input_17 => relu_8
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_26,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 728)
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
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = triton_helpers.maximum(tmp16, tmp15)
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/bl/cbltlo5qb6jlhbdh3wzz5t7afddcdmvy4g62h3dcc4tu4b7rjxjn.py
# Topologically Sorted Source Nodes: [input_20, x_27, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_20 => add_30, mul_40, mul_41, sub_13
#   input_21 => relu_10
#   x_27 => add_31
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
#   %add_31 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %add_24), kwargs = {})
#   %relu_10 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_31,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 46592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 728)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_out_ptr0 + (x2), xmask)
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
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tl.store(in_out_ptr0 + (x2), tmp17, xmask)
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/fe/cfeciymi23jwoiwmxc7uukdpbg5ihjy7pximyupt4uijp5udsnqt.py
# Topologically Sorted Source Nodes: [input_66], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   input_66 => add_84, mul_109, mul_110, sub_36
# Graph fragment:
#   %sub_36 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_68, %unsqueeze_289), kwargs = {})
#   %mul_109 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_36, %unsqueeze_291), kwargs = {})
#   %mul_110 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_109, %unsqueeze_293), kwargs = {})
#   %add_84 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_110, %unsqueeze_295), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_16 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_16', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1024)
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/t5/ct5qsixtfv4v4gtfjjrndi7cbksqkzddpjv4wcikmeqatioapysn.py
# Topologically Sorted Source Nodes: [input_67, skip_7, x_81], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add]
# Source node to ATen node mapping:
#   input_67 => _low_memory_max_pool2d_with_offsets_3, getitem_7
#   skip_7 => add_86, mul_112, mul_113, sub_37
#   x_81 => add_87
# Graph fragment:
#   %_low_memory_max_pool2d_with_offsets_3 : [num_users=2] = call_function[target=torch.ops.prims._low_memory_max_pool2d_with_offsets.default](args = (%add_84, [3, 3], [2, 2], [1, 1], [1, 1], False), kwargs = {})
#   %getitem_7 : [num_users=1] = call_function[target=operator.getitem](args = (%_low_memory_max_pool2d_with_offsets_3, 1), kwargs = {})
#   %sub_37 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_69, %unsqueeze_297), kwargs = {})
#   %mul_112 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_37, %unsqueeze_299), kwargs = {})
#   %mul_113 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_112, %unsqueeze_301), kwargs = {})
#   %add_86 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_113, %unsqueeze_303), kwargs = {})
#   %add_87 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, %add_86), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i8', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 14, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = ((xindex // 2048) % 2)
    x1 = ((xindex // 1024) % 2)
    x0 = (xindex % 1024)
    x6 = xindex // 2048
    x7 = xindex
    tmp77 = tl.load(in_ptr1 + (x7), None)
    tmp78 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp89 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp91 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 4, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-5120) + x0 + 2048*x1 + 8192*x6), tmp10, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-4096) + x0 + 2048*x1 + 8192*x6), tmp16, other=float("-inf"))
    tmp18 = triton_helpers.maximum(tmp17, tmp11)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-3072) + x0 + 2048*x1 + 8192*x6), tmp23, other=float("-inf"))
    tmp25 = triton_helpers.maximum(tmp24, tmp18)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1024) + x0 + 2048*x1 + 8192*x6), tmp30, other=float("-inf"))
    tmp32 = triton_helpers.maximum(tmp31, tmp25)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 2048*x1 + 8192*x6), tmp33, other=float("-inf"))
    tmp35 = triton_helpers.maximum(tmp34, tmp32)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1024 + x0 + 2048*x1 + 8192*x6), tmp36, other=float("-inf"))
    tmp38 = triton_helpers.maximum(tmp37, tmp35)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (3072 + x0 + 2048*x1 + 8192*x6), tmp43, other=float("-inf"))
    tmp45 = triton_helpers.maximum(tmp44, tmp38)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (4096 + x0 + 2048*x1 + 8192*x6), tmp46, other=float("-inf"))
    tmp48 = triton_helpers.maximum(tmp47, tmp45)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (5120 + x0 + 2048*x1 + 8192*x6), tmp49, other=float("-inf"))
    tmp51 = triton_helpers.maximum(tmp50, tmp48)
    tmp52 = tmp17 > tmp11
    tmp53 = tl.full([1], 1, tl.int8)
    tmp54 = tl.full([1], 0, tl.int8)
    tmp55 = tl.where(tmp52, tmp53, tmp54)
    tmp56 = tmp24 > tmp18
    tmp57 = tl.full([1], 2, tl.int8)
    tmp58 = tl.where(tmp56, tmp57, tmp55)
    tmp59 = tmp31 > tmp25
    tmp60 = tl.full([1], 3, tl.int8)
    tmp61 = tl.where(tmp59, tmp60, tmp58)
    tmp62 = tmp34 > tmp32
    tmp63 = tl.full([1], 4, tl.int8)
    tmp64 = tl.where(tmp62, tmp63, tmp61)
    tmp65 = tmp37 > tmp35
    tmp66 = tl.full([1], 5, tl.int8)
    tmp67 = tl.where(tmp65, tmp66, tmp64)
    tmp68 = tmp44 > tmp38
    tmp69 = tl.full([1], 6, tl.int8)
    tmp70 = tl.where(tmp68, tmp69, tmp67)
    tmp71 = tmp47 > tmp45
    tmp72 = tl.full([1], 7, tl.int8)
    tmp73 = tl.where(tmp71, tmp72, tmp70)
    tmp74 = tmp50 > tmp48
    tmp75 = tl.full([1], 8, tl.int8)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp79 = tmp77 - tmp78
    tmp81 = 1e-05
    tmp82 = tmp80 + tmp81
    tmp83 = libdevice.sqrt(tmp82)
    tmp84 = tl.full([1], 1, tl.int32)
    tmp85 = tmp84 / tmp83
    tmp86 = 1.0
    tmp87 = tmp85 * tmp86
    tmp88 = tmp79 * tmp87
    tmp90 = tmp88 * tmp89
    tmp92 = tmp90 + tmp91
    tmp93 = tmp51 + tmp92
    tl.store(out_ptr0 + (x7), tmp76, None)
    tl.store(in_out_ptr0 + (x7), tmp93, None)
''', device_str='cuda')


# kernel path: inductor_cache/fp/cfpvwkrdqz6qf4pl3vzx5uoggryhcggbr4tyvlapwgojspqtf7hx.py
# Topologically Sorted Source Nodes: [x_84, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_84 => add_89, mul_115, mul_116, sub_38
#   x_85 => relu_33
# Graph fragment:
#   %sub_38 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_71, %unsqueeze_305), kwargs = {})
#   %mul_115 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_38, %unsqueeze_307), kwargs = {})
#   %mul_116 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_115, %unsqueeze_309), kwargs = {})
#   %add_89 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_116, %unsqueeze_311), kwargs = {})
#   %relu_33 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_89,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 1536)
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


# kernel path: inductor_cache/xq/cxqnwo4v53lu2tina4s6q3blpejc2wr5n4cqc7fxwm2h46z4xoi3.py
# Topologically Sorted Source Nodes: [x_88, x_89, v], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   v => mean
#   x_88 => add_91, mul_118, mul_119, sub_39
#   x_89 => relu_34
# Graph fragment:
#   %sub_39 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_73, %unsqueeze_313), kwargs = {})
#   %mul_118 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_39, %unsqueeze_315), kwargs = {})
#   %mul_119 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_118, %unsqueeze_317), kwargs = {})
#   %add_91 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_119, %unsqueeze_319), kwargs = {})
#   %relu_34 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_91,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_34, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_19', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 2048)
    x1 = xindex // 2048
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 8192*x1), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr0 + (2048 + x0 + 8192*x1), None)
    tmp25 = tl.load(in_ptr0 + (4096 + x0 + 8192*x1), None)
    tmp32 = tl.load(in_ptr0 + (6144 + x0 + 8192*x1), None)
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
    tmp19 = tmp18 - tmp1
    tmp20 = tmp19 * tmp10
    tmp21 = tmp20 * tmp12
    tmp22 = tmp21 + tmp14
    tmp23 = triton_helpers.maximum(tmp16, tmp22)
    tmp24 = tmp17 + tmp23
    tmp26 = tmp25 - tmp1
    tmp27 = tmp26 * tmp10
    tmp28 = tmp27 * tmp12
    tmp29 = tmp28 + tmp14
    tmp30 = triton_helpers.maximum(tmp16, tmp29)
    tmp31 = tmp24 + tmp30
    tmp33 = tmp32 - tmp1
    tmp34 = tmp33 * tmp10
    tmp35 = tmp34 * tmp12
    tmp36 = tmp35 + tmp14
    tmp37 = triton_helpers.maximum(tmp16, tmp36)
    tmp38 = tmp31 + tmp37
    tmp39 = 4.0
    tmp40 = tmp38 / tmp39
    tl.store(out_ptr0 + (x2), tmp40, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235 = args
    args.clear()
    assert_size_stride(primals_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (4, 3, 64, 64), (12288, 4096, 64, 1))
    assert_size_stride(primals_3, (32, ), (1, ))
    assert_size_stride(primals_4, (32, ), (1, ))
    assert_size_stride(primals_5, (32, ), (1, ))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (64, ), (1, ))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_13, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_19, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, ), (1, ))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_30, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_31, (256, ), (1, ))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_36, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_42, (256, ), (1, ))
    assert_size_stride(primals_43, (256, ), (1, ))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (256, ), (1, ))
    assert_size_stride(primals_46, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_47, (728, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_48, (728, ), (1, ))
    assert_size_stride(primals_49, (728, ), (1, ))
    assert_size_stride(primals_50, (728, ), (1, ))
    assert_size_stride(primals_51, (728, ), (1, ))
    assert_size_stride(primals_52, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_53, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_54, (728, ), (1, ))
    assert_size_stride(primals_55, (728, ), (1, ))
    assert_size_stride(primals_56, (728, ), (1, ))
    assert_size_stride(primals_57, (728, ), (1, ))
    assert_size_stride(primals_58, (728, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (728, ), (1, ))
    assert_size_stride(primals_60, (728, ), (1, ))
    assert_size_stride(primals_61, (728, ), (1, ))
    assert_size_stride(primals_62, (728, ), (1, ))
    assert_size_stride(primals_63, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_64, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_65, (728, ), (1, ))
    assert_size_stride(primals_66, (728, ), (1, ))
    assert_size_stride(primals_67, (728, ), (1, ))
    assert_size_stride(primals_68, (728, ), (1, ))
    assert_size_stride(primals_69, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_70, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_71, (728, ), (1, ))
    assert_size_stride(primals_72, (728, ), (1, ))
    assert_size_stride(primals_73, (728, ), (1, ))
    assert_size_stride(primals_74, (728, ), (1, ))
    assert_size_stride(primals_75, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_76, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_77, (728, ), (1, ))
    assert_size_stride(primals_78, (728, ), (1, ))
    assert_size_stride(primals_79, (728, ), (1, ))
    assert_size_stride(primals_80, (728, ), (1, ))
    assert_size_stride(primals_81, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_82, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_83, (728, ), (1, ))
    assert_size_stride(primals_84, (728, ), (1, ))
    assert_size_stride(primals_85, (728, ), (1, ))
    assert_size_stride(primals_86, (728, ), (1, ))
    assert_size_stride(primals_87, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_88, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_89, (728, ), (1, ))
    assert_size_stride(primals_90, (728, ), (1, ))
    assert_size_stride(primals_91, (728, ), (1, ))
    assert_size_stride(primals_92, (728, ), (1, ))
    assert_size_stride(primals_93, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_94, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_95, (728, ), (1, ))
    assert_size_stride(primals_96, (728, ), (1, ))
    assert_size_stride(primals_97, (728, ), (1, ))
    assert_size_stride(primals_98, (728, ), (1, ))
    assert_size_stride(primals_99, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_100, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_101, (728, ), (1, ))
    assert_size_stride(primals_102, (728, ), (1, ))
    assert_size_stride(primals_103, (728, ), (1, ))
    assert_size_stride(primals_104, (728, ), (1, ))
    assert_size_stride(primals_105, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_106, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_107, (728, ), (1, ))
    assert_size_stride(primals_108, (728, ), (1, ))
    assert_size_stride(primals_109, (728, ), (1, ))
    assert_size_stride(primals_110, (728, ), (1, ))
    assert_size_stride(primals_111, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_112, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_113, (728, ), (1, ))
    assert_size_stride(primals_114, (728, ), (1, ))
    assert_size_stride(primals_115, (728, ), (1, ))
    assert_size_stride(primals_116, (728, ), (1, ))
    assert_size_stride(primals_117, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_119, (728, ), (1, ))
    assert_size_stride(primals_120, (728, ), (1, ))
    assert_size_stride(primals_121, (728, ), (1, ))
    assert_size_stride(primals_122, (728, ), (1, ))
    assert_size_stride(primals_123, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_124, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_125, (728, ), (1, ))
    assert_size_stride(primals_126, (728, ), (1, ))
    assert_size_stride(primals_127, (728, ), (1, ))
    assert_size_stride(primals_128, (728, ), (1, ))
    assert_size_stride(primals_129, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_130, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_131, (728, ), (1, ))
    assert_size_stride(primals_132, (728, ), (1, ))
    assert_size_stride(primals_133, (728, ), (1, ))
    assert_size_stride(primals_134, (728, ), (1, ))
    assert_size_stride(primals_135, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_136, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_137, (728, ), (1, ))
    assert_size_stride(primals_138, (728, ), (1, ))
    assert_size_stride(primals_139, (728, ), (1, ))
    assert_size_stride(primals_140, (728, ), (1, ))
    assert_size_stride(primals_141, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_142, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_143, (728, ), (1, ))
    assert_size_stride(primals_144, (728, ), (1, ))
    assert_size_stride(primals_145, (728, ), (1, ))
    assert_size_stride(primals_146, (728, ), (1, ))
    assert_size_stride(primals_147, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_148, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_149, (728, ), (1, ))
    assert_size_stride(primals_150, (728, ), (1, ))
    assert_size_stride(primals_151, (728, ), (1, ))
    assert_size_stride(primals_152, (728, ), (1, ))
    assert_size_stride(primals_153, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_154, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_155, (728, ), (1, ))
    assert_size_stride(primals_156, (728, ), (1, ))
    assert_size_stride(primals_157, (728, ), (1, ))
    assert_size_stride(primals_158, (728, ), (1, ))
    assert_size_stride(primals_159, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_160, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_161, (728, ), (1, ))
    assert_size_stride(primals_162, (728, ), (1, ))
    assert_size_stride(primals_163, (728, ), (1, ))
    assert_size_stride(primals_164, (728, ), (1, ))
    assert_size_stride(primals_165, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_166, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_167, (728, ), (1, ))
    assert_size_stride(primals_168, (728, ), (1, ))
    assert_size_stride(primals_169, (728, ), (1, ))
    assert_size_stride(primals_170, (728, ), (1, ))
    assert_size_stride(primals_171, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_172, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_173, (728, ), (1, ))
    assert_size_stride(primals_174, (728, ), (1, ))
    assert_size_stride(primals_175, (728, ), (1, ))
    assert_size_stride(primals_176, (728, ), (1, ))
    assert_size_stride(primals_177, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_178, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_179, (728, ), (1, ))
    assert_size_stride(primals_180, (728, ), (1, ))
    assert_size_stride(primals_181, (728, ), (1, ))
    assert_size_stride(primals_182, (728, ), (1, ))
    assert_size_stride(primals_183, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_184, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_185, (728, ), (1, ))
    assert_size_stride(primals_186, (728, ), (1, ))
    assert_size_stride(primals_187, (728, ), (1, ))
    assert_size_stride(primals_188, (728, ), (1, ))
    assert_size_stride(primals_189, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_190, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_191, (728, ), (1, ))
    assert_size_stride(primals_192, (728, ), (1, ))
    assert_size_stride(primals_193, (728, ), (1, ))
    assert_size_stride(primals_194, (728, ), (1, ))
    assert_size_stride(primals_195, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_196, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_197, (728, ), (1, ))
    assert_size_stride(primals_198, (728, ), (1, ))
    assert_size_stride(primals_199, (728, ), (1, ))
    assert_size_stride(primals_200, (728, ), (1, ))
    assert_size_stride(primals_201, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_202, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_203, (728, ), (1, ))
    assert_size_stride(primals_204, (728, ), (1, ))
    assert_size_stride(primals_205, (728, ), (1, ))
    assert_size_stride(primals_206, (728, ), (1, ))
    assert_size_stride(primals_207, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_208, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_209, (728, ), (1, ))
    assert_size_stride(primals_210, (728, ), (1, ))
    assert_size_stride(primals_211, (728, ), (1, ))
    assert_size_stride(primals_212, (728, ), (1, ))
    assert_size_stride(primals_213, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_214, (1024, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_215, (1024, ), (1, ))
    assert_size_stride(primals_216, (1024, ), (1, ))
    assert_size_stride(primals_217, (1024, ), (1, ))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (1024, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_220, (1024, ), (1, ))
    assert_size_stride(primals_221, (1024, ), (1, ))
    assert_size_stride(primals_222, (1024, ), (1, ))
    assert_size_stride(primals_223, (1024, ), (1, ))
    assert_size_stride(primals_224, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_225, (1536, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_226, (1536, ), (1, ))
    assert_size_stride(primals_227, (1536, ), (1, ))
    assert_size_stride(primals_228, (1536, ), (1, ))
    assert_size_stride(primals_229, (1536, ), (1, ))
    assert_size_stride(primals_230, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_231, (2048, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_232, (2048, ), (1, ))
    assert_size_stride(primals_233, (2048, ), (1, ))
    assert_size_stride(primals_234, (2048, ), (1, ))
    assert_size_stride(primals_235, (2048, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((32, 3, 3, 3), (27, 1, 9, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 96, 9, grid=grid(96, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided_cuda((4, 3, 64, 64), (12288, 1, 192, 3), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_1.run(primals_2, buf1, 12, 4096, grid=grid(12, 4096), stream=stream0)
        del primals_2
        buf2 = empty_strided_cuda((64, 32, 3, 3), (288, 1, 96, 32), torch.float32)
        # Unsorted Source Nodes: [], Original ATen: []
        stream0 = get_raw_stream(0)
        triton_poi_fused_2.run(primals_7, buf2, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del primals_7
        # Topologically Sorted Source Nodes: [x], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 32, 31, 31), (30752, 1, 992, 32))
        buf4 = empty_strided_cuda((4, 32, 31, 31), (30752, 1, 992, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf3, primals_3, primals_4, primals_5, primals_6, buf4, 123008, grid=grid(123008), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, buf2, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 29, 29), (53824, 1, 1856, 64))
        buf6 = empty_strided_cuda((4, 64, 29, 29), (53824, 1, 1856, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf5, primals_8, primals_9, primals_10, primals_11, buf6, 215296, grid=grid(215296), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf7, (4, 64, 29, 29), (53824, 1, 1856, 64))
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 128, 29, 29), (107648, 1, 3712, 128))
        buf9 = empty_strided_cuda((4, 128, 29, 29), (107648, 1, 3712, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf8, primals_14, primals_15, primals_16, primals_17, buf9, 430592, grid=grid(430592), stream=stream0)
        del primals_17
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf10, (4, 128, 29, 29), (107648, 1, 3712, 128))
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 29, 29), (107648, 1, 3712, 128))
        buf12 = empty_strided_cuda((4, 128, 29, 29), (107648, 1, 3712, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_3], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf11, primals_20, primals_21, primals_22, primals_23, buf12, 430592, grid=grid(430592), stream=stream0)
        del primals_23
        # Topologically Sorted Source Nodes: [skip], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf6, primals_24, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 128, 15, 15), (28800, 1, 1920, 128))
        buf13 = empty_strided_cuda((4, 128, 15, 15), (28800, 1, 1920, 128), torch.float32)
        buf14 = empty_strided_cuda((4, 128, 15, 15), (28800, 1, 1920, 128), torch.int8)
        buf16 = buf13; del buf13  # reuse
        buf17 = empty_strided_cuda((4, 128, 15, 15), (28800, 1, 1920, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_4, skip_1, x_10, input_5], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_7.run(buf16, buf12, buf15, primals_25, primals_26, primals_27, primals_28, buf14, buf17, 115200, grid=grid(115200), stream=stream0)
        del primals_28
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_29, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf18, (4, 128, 15, 15), (28800, 1, 1920, 128))
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 15, 15), (57600, 1, 3840, 256))
        buf20 = empty_strided_cuda((4, 256, 15, 15), (57600, 1, 3840, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf19, primals_31, primals_32, primals_33, primals_34, buf20, 230400, grid=grid(230400), stream=stream0)
        del primals_34
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf21, (4, 256, 15, 15), (57600, 1, 3840, 256))
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 15, 15), (57600, 1, 3840, 256))
        buf23 = empty_strided_cuda((4, 256, 15, 15), (57600, 1, 3840, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_8], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_9.run(buf22, primals_37, primals_38, primals_39, primals_40, buf23, 230400, grid=grid(230400), stream=stream0)
        del primals_40
        # Topologically Sorted Source Nodes: [skip_2], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf16, primals_41, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf24 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf25 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.int8)
        buf27 = buf24; del buf24  # reuse
        buf28 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_9, skip_3, x_15, input_10], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_10.run(buf27, buf23, buf26, primals_42, primals_43, primals_44, primals_45, buf25, buf28, 65536, grid=grid(65536), stream=stream0)
        del primals_45
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf29, (4, 256, 8, 8), (16384, 1, 2048, 256))
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 728, 8, 8), (46592, 1, 5824, 728))
        buf31 = empty_strided_cuda((4, 728, 8, 8), (46592, 1, 5824, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_11, input_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf30, primals_48, primals_49, primals_50, primals_51, buf31, 186368, grid=grid(186368), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf32, (4, 728, 8, 8), (46592, 1, 5824, 728))
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 728, 8, 8), (46592, 1, 5824, 728))
        buf34 = empty_strided_cuda((4, 728, 8, 8), (46592, 1, 5824, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_13], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf33, primals_54, primals_55, primals_56, primals_57, buf34, 186368, grid=grid(186368), stream=stream0)
        del primals_57
        # Topologically Sorted Source Nodes: [skip_4], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf27, primals_58, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf35 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        buf36 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.int8)
        buf38 = buf35; del buf35  # reuse
        buf39 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, skip_5, x_20, input_15], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_relu_13.run(buf38, buf34, buf37, primals_59, primals_60, primals_61, primals_62, buf36, buf39, 46592, grid=grid(46592), stream=stream0)
        del primals_62
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_63, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf40, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf42 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_16, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf41, primals_65, primals_66, primals_67, primals_68, buf42, 46592, grid=grid(46592), stream=stream0)
        del primals_68
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_69, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf43, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf45 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf44, primals_71, primals_72, primals_73, primals_74, buf45, 46592, grid=grid(46592), stream=stream0)
        del primals_74
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_75, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf46, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf48 = buf38; del buf38  # reuse
        buf49 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_20, x_27, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf48, buf47, primals_77, primals_78, primals_79, primals_80, buf49, 46592, grid=grid(46592), stream=stream0)
        del primals_80
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf50, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf52 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf51, primals_83, primals_84, primals_85, primals_86, buf52, 46592, grid=grid(46592), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_87, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf53, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf55 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf54, primals_89, primals_90, primals_91, primals_92, buf55, 46592, grid=grid(46592), stream=stream0)
        del primals_92
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, primals_93, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf56, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf58 = buf48; del buf48  # reuse
        buf59 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, x_34, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf58, buf57, primals_95, primals_96, primals_97, primals_98, buf59, 46592, grid=grid(46592), stream=stream0)
        del primals_98
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_99, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf60, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf62 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf61, primals_101, primals_102, primals_103, primals_104, buf62, 46592, grid=grid(46592), stream=stream0)
        del primals_104
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_105, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf63, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf65 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf64, primals_107, primals_108, primals_109, primals_110, buf65, 46592, grid=grid(46592), stream=stream0)
        del primals_110
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_111, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf66, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf68 = buf58; del buf58  # reuse
        buf69 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, x_41, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf68, buf67, primals_113, primals_114, primals_115, primals_116, buf69, 46592, grid=grid(46592), stream=stream0)
        del primals_116
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf70, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf72 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, input_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf71, primals_119, primals_120, primals_121, primals_122, buf72, 46592, grid=grid(46592), stream=stream0)
        del primals_122
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf73, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_124, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf75 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf74, primals_125, primals_126, primals_127, primals_128, buf75, 46592, grid=grid(46592), stream=stream0)
        del primals_128
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf76, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf78 = buf68; del buf68  # reuse
        buf79 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, x_48, input_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf78, buf77, primals_131, primals_132, primals_133, primals_134, buf79, 46592, grid=grid(46592), stream=stream0)
        del primals_134
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf80, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf82 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_40, input_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf81, primals_137, primals_138, primals_139, primals_140, buf82, 46592, grid=grid(46592), stream=stream0)
        del primals_140
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf83, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf85 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf84, primals_143, primals_144, primals_145, primals_146, buf85, 46592, grid=grid(46592), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf86, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf88 = buf78; del buf78  # reuse
        buf89 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, x_55, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf88, buf87, primals_149, primals_150, primals_151, primals_152, buf89, 46592, grid=grid(46592), stream=stream0)
        del primals_152
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf90, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf92 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, input_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf91, primals_155, primals_156, primals_157, primals_158, buf92, 46592, grid=grid(46592), stream=stream0)
        del primals_158
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf93, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten.convolution]
        buf94 = extern_kernels.convolution(buf93, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf95 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf94, primals_161, primals_162, primals_163, primals_164, buf95, 46592, grid=grid(46592), stream=stream0)
        del primals_164
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf96, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf98 = buf88; del buf88  # reuse
        buf99 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, x_62, input_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf98, buf97, primals_167, primals_168, primals_169, primals_170, buf99, 46592, grid=grid(46592), stream=stream0)
        del primals_170
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf100, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf102 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_52, input_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf101, primals_173, primals_174, primals_175, primals_176, buf102, 46592, grid=grid(46592), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf103, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf105 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf104, primals_179, primals_180, primals_181, primals_182, buf105, 46592, grid=grid(46592), stream=stream0)
        del primals_182
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf106, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf108 = buf98; del buf98  # reuse
        buf109 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, x_69, input_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf108, buf107, primals_185, primals_186, primals_187, primals_188, buf109, 46592, grid=grid(46592), stream=stream0)
        del primals_188
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_189, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf110, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf112 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_58, input_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf111, primals_191, primals_192, primals_193, primals_194, buf112, 46592, grid=grid(46592), stream=stream0)
        del primals_194
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf113, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf115 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf114, primals_197, primals_198, primals_199, primals_200, buf115, 46592, grid=grid(46592), stream=stream0)
        del primals_200
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_201, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf116, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf118 = buf108; del buf108  # reuse
        buf119 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_62, x_76, input_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf118, buf117, primals_203, primals_204, primals_205, primals_206, buf119, 46592, grid=grid(46592), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf120, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf122 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_64, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf121, primals_209, primals_210, primals_211, primals_212, buf122, 46592, grid=grid(46592), stream=stream0)
        del primals_212
        # Topologically Sorted Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf123, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf125 = empty_strided_cuda((4, 1024, 4, 4), (16384, 1, 4096, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [input_66], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf124, primals_215, primals_216, primals_217, primals_218, buf125, 65536, grid=grid(65536), stream=stream0)
        del primals_218
        # Topologically Sorted Source Nodes: [skip_6], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf118, primals_219, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf126 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf127 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.int8)
        buf129 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [input_67, skip_7, x_81], Original ATen: [aten.max_pool2d_with_indices, aten._native_batch_norm_legit_no_training, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_max_pool2d_with_indices_17.run(buf129, buf125, buf128, primals_220, primals_221, primals_222, primals_223, buf127, 16384, grid=grid(16384), stream=stream0)
        del primals_223
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, primals_224, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf130, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        # Topologically Sorted Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf132 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_84, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf131, primals_226, primals_227, primals_228, primals_229, buf132, 24576, grid=grid(24576), stream=stream0)
        del primals_229
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf133, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        # Topologically Sorted Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf135 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_88, x_89, v], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf134, primals_232, primals_233, primals_234, primals_235, buf135, 8192, grid=grid(8192), stream=stream0)
    return (reinterpret_tensor(buf135, (4, 2048), (2048, 1), 0), buf0, buf1, primals_3, primals_4, primals_5, buf2, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_16, primals_18, primals_19, primals_20, primals_21, primals_22, primals_24, primals_25, primals_26, primals_27, primals_29, primals_30, primals_31, primals_32, primals_33, primals_35, primals_36, primals_37, primals_38, primals_39, primals_41, primals_42, primals_43, primals_44, primals_46, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_56, primals_58, primals_59, primals_60, primals_61, primals_63, primals_64, primals_65, primals_66, primals_67, primals_69, primals_70, primals_71, primals_72, primals_73, primals_75, primals_76, primals_77, primals_78, primals_79, primals_81, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_91, primals_93, primals_94, primals_95, primals_96, primals_97, primals_99, primals_100, primals_101, primals_102, primals_103, primals_105, primals_106, primals_107, primals_108, primals_109, primals_111, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_121, primals_123, primals_124, primals_125, primals_126, primals_127, primals_129, primals_130, primals_131, primals_132, primals_133, primals_135, primals_136, primals_137, primals_138, primals_139, primals_141, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_151, primals_153, primals_154, primals_155, primals_156, primals_157, primals_159, primals_160, primals_161, primals_162, primals_163, primals_165, primals_166, primals_167, primals_168, primals_169, primals_171, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_181, primals_183, primals_184, primals_185, primals_186, primals_187, primals_189, primals_190, primals_191, primals_192, primals_193, primals_195, primals_196, primals_197, primals_198, primals_199, primals_201, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_211, primals_213, primals_214, primals_215, primals_216, primals_217, primals_219, primals_220, primals_221, primals_222, primals_224, primals_225, primals_226, primals_227, primals_228, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf36, buf37, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 3, 64, 64), (12288, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((728, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((728, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1024, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1024, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((1536, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((2048, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
