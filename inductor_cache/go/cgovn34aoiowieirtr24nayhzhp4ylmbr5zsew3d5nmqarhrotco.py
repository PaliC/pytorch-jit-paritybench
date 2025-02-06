# AOT ID: ['33_forward']
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


# kernel path: inductor_cache/wh/cwhie2gvluoobdto4f7jfzc2mfczbebuowelsq3c3aofpyelqbyf.py
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
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 32)
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


# kernel path: inductor_cache/ux/cuxchtjneyogqnog7hnnsqygkjlclvwkv7iw6bjlo537bctzr2rk.py
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
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
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


# kernel path: inductor_cache/6w/c6wdqvhtjpvbtuwlrlqq65nt7zaefftiezrwhoxmbck6ncdgcuor.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_7 => add_5, mul_7, mul_8, sub_2
# Graph fragment:
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_2, %unsqueeze_17), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2, %unsqueeze_19), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_7, %unsqueeze_21), kwargs = {})
#   %add_5 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %unsqueeze_23), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_5 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_5', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_5', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 64)
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


# kernel path: inductor_cache/nt/cnttivw3q33mua2kera7j3deicq5fpo6v5ahc5kgymdiehl2chib.py
# Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_1 => add_7, mul_10, mul_11, sub_3
#   input_2 => relu_2
# Graph fragment:
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_3, %unsqueeze_25), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %unsqueeze_27), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %unsqueeze_29), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %unsqueeze_31), kwargs = {})
#   %relu_2 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_7,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_6', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
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


# kernel path: inductor_cache/cf/ccfonh6ijs2f5b5zzldtjjttnklvx3eiwoep6w74xdiwu5pp5r7x.py
# Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_10 => add_9, mul_13, mul_14, sub_4
# Graph fragment:
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_4, %unsqueeze_33), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %unsqueeze_35), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_13, %unsqueeze_37), kwargs = {})
#   %add_9 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_14, %unsqueeze_39), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_7', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
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


# kernel path: inductor_cache/en/cen5dirrywhtlmzgqhl5fptrkoxyplrf4zcjjemqpq5pfm3cpohm.py
# Topologically Sorted Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_13 => add_13, mul_19, mul_20, sub_6
# Graph fragment:
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_6, %unsqueeze_49), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_6, %unsqueeze_51), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_19, %unsqueeze_53), kwargs = {})
#   %add_13 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %unsqueeze_55), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_8 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_8', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
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


# kernel path: inductor_cache/nu/cnuf3v75lclsgraqe7ksc5crq3jmpn52ap7k3yxcz5jiz33zac6f.py
# Topologically Sorted Source Nodes: [input_5, skip, out, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_5 => add_15, mul_22, mul_23, sub_7
#   out => add_18
#   skip => add_17, mul_25, mul_26, sub_8
#   x_15 => relu_4
# Graph fragment:
#   %sub_7 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_7, %unsqueeze_57), kwargs = {})
#   %mul_22 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_7, %unsqueeze_59), kwargs = {})
#   %mul_23 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_22, %unsqueeze_61), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_23, %unsqueeze_63), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_8, %unsqueeze_65), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %unsqueeze_67), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_25, %unsqueeze_69), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_26, %unsqueeze_71), kwargs = {})
#   %add_18 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %add_17), kwargs = {})
#   %relu_4 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_18,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 128)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/27/c27i4yfk3p6ssy7dgqwdst767mmwmvsu6co5ebe6jxbt2tc3szsd.py
# Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_6 => add_22, mul_31, mul_32, sub_10
#   input_7 => relu_5
# Graph fragment:
#   %sub_10 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_10, %unsqueeze_81), kwargs = {})
#   %mul_31 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_10, %unsqueeze_83), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_31, %unsqueeze_85), kwargs = {})
#   %add_22 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_32, %unsqueeze_87), kwargs = {})
#   %relu_5 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_22,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_10', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
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


# kernel path: inductor_cache/dg/cdg5rnhh223sxdadewkyc3ns7ae3yskmvepv53af77uiokck6tkc.py
# Topologically Sorted Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_20 => add_24, mul_34, mul_35, sub_11
# Graph fragment:
#   %sub_11 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_11, %unsqueeze_89), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_11, %unsqueeze_91), kwargs = {})
#   %mul_35 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_34, %unsqueeze_93), kwargs = {})
#   %add_24 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_35, %unsqueeze_95), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_11 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_11', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_11(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
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


# kernel path: inductor_cache/sp/cspqccldvzocqqp3jguf7bisj4ybly3lpi7f7mb2opwmcfp2nadm.py
# Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_23 => add_28, mul_40, mul_41, sub_13
# Graph fragment:
#   %sub_13 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_13, %unsqueeze_105), kwargs = {})
#   %mul_40 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_13, %unsqueeze_107), kwargs = {})
#   %mul_41 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_40, %unsqueeze_109), kwargs = {})
#   %add_28 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_41, %unsqueeze_111), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_12 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_12', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
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


# kernel path: inductor_cache/yj/cyjog5kerci7y5wd4vj34gf4qbdnlwbb5ij4xfwd7si5u2ttk672.py
# Topologically Sorted Source Nodes: [input_10, skip_1, out_1, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_10 => add_30, mul_43, mul_44, sub_14
#   input_11 => relu_7
#   out_1 => add_33
#   skip_1 => add_32, mul_46, mul_47, sub_15
# Graph fragment:
#   %sub_14 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_14, %unsqueeze_113), kwargs = {})
#   %mul_43 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_14, %unsqueeze_115), kwargs = {})
#   %mul_44 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_43, %unsqueeze_117), kwargs = {})
#   %add_30 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_44, %unsqueeze_119), kwargs = {})
#   %sub_15 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_15, %unsqueeze_121), kwargs = {})
#   %mul_46 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_15, %unsqueeze_123), kwargs = {})
#   %mul_47 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_46, %unsqueeze_125), kwargs = {})
#   %add_32 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_47, %unsqueeze_127), kwargs = {})
#   %add_33 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %add_32), kwargs = {})
#   %relu_7 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_33,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 256)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/fe/cfecvd6cukfwnczzcgcr74z73pu5qqww3yip35s67kgbbgfa7aaz.py
# Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_12 => add_37, mul_52, mul_53, sub_17
#   input_13 => relu_8
# Graph fragment:
#   %sub_17 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_17, %unsqueeze_137), kwargs = {})
#   %mul_52 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_17, %unsqueeze_139), kwargs = {})
#   %mul_53 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_52, %unsqueeze_141), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_53, %unsqueeze_143), kwargs = {})
#   %relu_8 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_37,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_14', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/v3/cv3yuu5qqout3pbdz45nirfmkoh2uvuotmpk5q3p7wjojidzantf.py
# Topologically Sorted Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_29 => add_39, mul_55, mul_56, sub_18
# Graph fragment:
#   %sub_18 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_18, %unsqueeze_145), kwargs = {})
#   %mul_55 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_18, %unsqueeze_147), kwargs = {})
#   %mul_56 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_55, %unsqueeze_149), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_56, %unsqueeze_151), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_15', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_15(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/au/cauwfdpgnues5bjd7iou7wwpzy7uqiaxip2peyybjmfpynpel32c.py
# Topologically Sorted Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_32 => add_43, mul_61, mul_62, sub_20
# Graph fragment:
#   %sub_20 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_20, %unsqueeze_161), kwargs = {})
#   %mul_61 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_20, %unsqueeze_163), kwargs = {})
#   %mul_62 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_61, %unsqueeze_165), kwargs = {})
#   %add_43 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_62, %unsqueeze_167), kwargs = {})
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/it/citnj2qwa4amfo5a3h4kg7k2x6xtw2iaxjxq7vu476e7skheaadx.py
# Topologically Sorted Source Nodes: [input_16, skip_2, out_2, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_16 => add_45, mul_64, mul_65, sub_21
#   input_17 => relu_10
#   out_2 => add_48
#   skip_2 => add_47, mul_67, mul_68, sub_22
# Graph fragment:
#   %sub_21 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_21, %unsqueeze_169), kwargs = {})
#   %mul_64 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_21, %unsqueeze_171), kwargs = {})
#   %mul_65 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_64, %unsqueeze_173), kwargs = {})
#   %add_45 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_65, %unsqueeze_175), kwargs = {})
#   %sub_22 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_22, %unsqueeze_177), kwargs = {})
#   %mul_67 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_22, %unsqueeze_179), kwargs = {})
#   %mul_68 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_67, %unsqueeze_181), kwargs = {})
#   %add_47 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_68, %unsqueeze_183), kwargs = {})
#   %add_48 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %add_47), kwargs = {})
#   %relu_10 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_48,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
    tmp17 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/zl/czlxal4rveu6oy2wtzrx55aly2o5n3k6ujtssblqf7viybuz4qwl.py
# Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   input_18 => add_52, mul_73, mul_74, sub_24
#   input_19 => relu_11
# Graph fragment:
#   %sub_24 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_24, %unsqueeze_193), kwargs = {})
#   %mul_73 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_24, %unsqueeze_195), kwargs = {})
#   %mul_74 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_73, %unsqueeze_197), kwargs = {})
#   %add_52 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_74, %unsqueeze_199), kwargs = {})
#   %relu_11 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_52,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_18', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/vq/cvqt3unjhem2vjfam26zobaiypz7xjqwyt7prs5cmveavipkw6c5.py
# Topologically Sorted Source Nodes: [input_22, out_3, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_22 => add_60, mul_85, mul_86, sub_28
#   input_23 => relu_13
#   out_3 => add_61
# Graph fragment:
#   %sub_28 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_28, %unsqueeze_225), kwargs = {})
#   %mul_85 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_28, %unsqueeze_227), kwargs = {})
#   %mul_86 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_85, %unsqueeze_229), kwargs = {})
#   %add_60 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_86, %unsqueeze_231), kwargs = {})
#   %add_61 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_60, %relu_10), kwargs = {})
#   %relu_13 : [num_users=3] = call_function[target=torch.ops.aten.relu.default](args = (%add_61,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.load(in_ptr5 + (x2), xmask)
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
    tl.store(out_ptr0 + (x2), tmp19, xmask)
''', device_str='cuda')


# kernel path: inductor_cache/tc/ctc3svsk6yii5ol3mn2kvuyyrxuckhw5rfqvuta7g2umta3bt5ps.py
# Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.relu]
# Source node to ATen node mapping:
#   input_116 => relu_60
# Graph fragment:
#   %relu_60 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%convolution_122,), kwargs = {})
triton_poi_fused_relu_20 = async_compile.triton('triton_poi_fused_relu_20', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_20', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_relu_20(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tl.store(in_out_ptr0 + (x0), tmp2, None)
''', device_str='cuda')


# kernel path: inductor_cache/6z/c6zpdymcehtyqtk4dlvxnf6nfsugftuvyuotboeisttzn63xjiye.py
# Topologically Sorted Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_185 => add_264, mul_367, mul_368, sub_122
# Graph fragment:
#   %sub_122 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_123, %unsqueeze_977), kwargs = {})
#   %mul_367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_122, %unsqueeze_979), kwargs = {})
#   %mul_368 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_367, %unsqueeze_981), kwargs = {})
#   %add_264 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_368, %unsqueeze_983), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_21', '''
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
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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


# kernel path: inductor_cache/py/cpy5danlrx2vho2tchqbjlgub3zgvisp6eqedgqd4uxiv67qsqio.py
# Topologically Sorted Source Nodes: [input_117, skip_3, out_19, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# Source node to ATen node mapping:
#   input_117 => add_266, mul_370, mul_371, sub_123
#   out_19 => add_269
#   skip_3 => add_268, mul_373, mul_374, sub_124
#   x_187 => relu_61
# Graph fragment:
#   %sub_123 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_124, %unsqueeze_985), kwargs = {})
#   %mul_370 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_123, %unsqueeze_987), kwargs = {})
#   %mul_371 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_370, %unsqueeze_989), kwargs = {})
#   %add_266 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_371, %unsqueeze_991), kwargs = {})
#   %sub_124 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_125, %unsqueeze_993), kwargs = {})
#   %mul_373 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_124, %unsqueeze_995), kwargs = {})
#   %mul_374 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_373, %unsqueeze_997), kwargs = {})
#   %add_268 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_374, %unsqueeze_999), kwargs = {})
#   %add_269 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_266, %add_268), kwargs = {})
#   %relu_61 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_269,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'in_ptr7': '*fp32', 'in_ptr8': '*fp32', 'in_ptr9': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor.from_dict({'arg_properties': {'tt.divisibility': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 'tt.equal_to': ()}, 'cls': 'AttrsDescriptor'})]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
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
    tmp16 = tl.load(in_ptr5 + (x2), None)
    tmp17 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
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
    tmp18 = tmp16 - tmp17
    tmp20 = tmp19 + tmp4
    tmp21 = libdevice.sqrt(tmp20)
    tmp22 = tmp7 / tmp21
    tmp23 = tmp22 * tmp9
    tmp24 = tmp18 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp15 + tmp28
    tmp30 = tl.full([1], 0, tl.int32)
    tmp31 = triton_helpers.maximum(tmp30, tmp29)
    tl.store(in_out_ptr0 + (x2), tmp31, None)
''', device_str='cuda')


# kernel path: inductor_cache/ne/cneiku2sk4foaqek4i3ve66hfvhh7ysjc3i5osxfgctkdbvqnpzo.py
# Topologically Sorted Source Nodes: [x_191, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# Source node to ATen node mapping:
#   x_191 => add_273, mul_379, mul_380, sub_126
#   x_192 => relu_62
# Graph fragment:
#   %sub_126 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_127, %unsqueeze_1009), kwargs = {})
#   %mul_379 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_126, %unsqueeze_1011), kwargs = {})
#   %mul_380 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_379, %unsqueeze_1013), kwargs = {})
#   %add_273 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_380, %unsqueeze_1015), kwargs = {})
#   %relu_62 : [num_users=2] = call_function[target=torch.ops.aten.relu.default](args = (%add_273,), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_relu_23', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_relu_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: inductor_cache/6j/c6jgkstubpaycufqyteqlflgyntswl2cx3yale6pdfhqrnohmxk2.py
# Topologically Sorted Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_no_training]
# Source node to ATen node mapping:
#   x_194 => add_275, mul_382, mul_383, sub_127
# Graph fragment:
#   %sub_127 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_128, %unsqueeze_1017), kwargs = {})
#   %mul_382 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_127, %unsqueeze_1019), kwargs = {})
#   %mul_383 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_382, %unsqueeze_1021), kwargs = {})
#   %add_275 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_383, %unsqueeze_1023), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_24 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_24', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: inductor_cache/an/canbxeg5k4tg4ubanebuc3dxbm7grrcduxfxgvens7co7de2elgp.py
# Topologically Sorted Source Nodes: [x_201, x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
# Source node to ATen node mapping:
#   x_201 => add_281, mul_391, mul_392, sub_130
#   x_202 => relu_64
#   x_203 => mean
# Graph fragment:
#   %sub_130 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convolution_131, %unsqueeze_1041), kwargs = {})
#   %mul_391 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_130, %unsqueeze_1043), kwargs = {})
#   %mul_392 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_391, %unsqueeze_1045), kwargs = {})
#   %add_281 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_392, %unsqueeze_1047), kwargs = {})
#   %relu_64 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_281,), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%relu_64, [-1, -2], True), kwargs = {})
triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_25 = async_compile.triton('triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_25', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': 'A0D3A2B50857E9501D843044B01F725922648D76E6D26323B14F8A4EA4473D1B', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659 = args
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
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (64, ), (1, ))
    assert_size_stride(primals_16, (64, ), (1, ))
    assert_size_stride(primals_17, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_28, (128, ), (1, ))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, ), (1, ))
    assert_size_stride(primals_35, (128, ), (1, ))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (128, ), (1, ))
    assert_size_stride(primals_41, (128, ), (1, ))
    assert_size_stride(primals_42, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (256, ), (1, ))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (256, ), (1, ))
    assert_size_stride(primals_62, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_73, (256, ), (1, ))
    assert_size_stride(primals_74, (256, ), (1, ))
    assert_size_stride(primals_75, (256, ), (1, ))
    assert_size_stride(primals_76, (256, ), (1, ))
    assert_size_stride(primals_77, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_78, (256, ), (1, ))
    assert_size_stride(primals_79, (256, ), (1, ))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (256, ), (1, ))
    assert_size_stride(primals_85, (256, ), (1, ))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (728, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_88, (728, ), (1, ))
    assert_size_stride(primals_89, (728, ), (1, ))
    assert_size_stride(primals_90, (728, ), (1, ))
    assert_size_stride(primals_91, (728, ), (1, ))
    assert_size_stride(primals_92, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_93, (728, ), (1, ))
    assert_size_stride(primals_94, (728, ), (1, ))
    assert_size_stride(primals_95, (728, ), (1, ))
    assert_size_stride(primals_96, (728, ), (1, ))
    assert_size_stride(primals_97, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_98, (728, ), (1, ))
    assert_size_stride(primals_99, (728, ), (1, ))
    assert_size_stride(primals_100, (728, ), (1, ))
    assert_size_stride(primals_101, (728, ), (1, ))
    assert_size_stride(primals_102, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_103, (728, ), (1, ))
    assert_size_stride(primals_104, (728, ), (1, ))
    assert_size_stride(primals_105, (728, ), (1, ))
    assert_size_stride(primals_106, (728, ), (1, ))
    assert_size_stride(primals_107, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_108, (728, ), (1, ))
    assert_size_stride(primals_109, (728, ), (1, ))
    assert_size_stride(primals_110, (728, ), (1, ))
    assert_size_stride(primals_111, (728, ), (1, ))
    assert_size_stride(primals_112, (728, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (728, ), (1, ))
    assert_size_stride(primals_114, (728, ), (1, ))
    assert_size_stride(primals_115, (728, ), (1, ))
    assert_size_stride(primals_116, (728, ), (1, ))
    assert_size_stride(primals_117, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_118, (728, ), (1, ))
    assert_size_stride(primals_119, (728, ), (1, ))
    assert_size_stride(primals_120, (728, ), (1, ))
    assert_size_stride(primals_121, (728, ), (1, ))
    assert_size_stride(primals_122, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_123, (728, ), (1, ))
    assert_size_stride(primals_124, (728, ), (1, ))
    assert_size_stride(primals_125, (728, ), (1, ))
    assert_size_stride(primals_126, (728, ), (1, ))
    assert_size_stride(primals_127, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_128, (728, ), (1, ))
    assert_size_stride(primals_129, (728, ), (1, ))
    assert_size_stride(primals_130, (728, ), (1, ))
    assert_size_stride(primals_131, (728, ), (1, ))
    assert_size_stride(primals_132, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_133, (728, ), (1, ))
    assert_size_stride(primals_134, (728, ), (1, ))
    assert_size_stride(primals_135, (728, ), (1, ))
    assert_size_stride(primals_136, (728, ), (1, ))
    assert_size_stride(primals_137, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_138, (728, ), (1, ))
    assert_size_stride(primals_139, (728, ), (1, ))
    assert_size_stride(primals_140, (728, ), (1, ))
    assert_size_stride(primals_141, (728, ), (1, ))
    assert_size_stride(primals_142, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_143, (728, ), (1, ))
    assert_size_stride(primals_144, (728, ), (1, ))
    assert_size_stride(primals_145, (728, ), (1, ))
    assert_size_stride(primals_146, (728, ), (1, ))
    assert_size_stride(primals_147, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_148, (728, ), (1, ))
    assert_size_stride(primals_149, (728, ), (1, ))
    assert_size_stride(primals_150, (728, ), (1, ))
    assert_size_stride(primals_151, (728, ), (1, ))
    assert_size_stride(primals_152, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_153, (728, ), (1, ))
    assert_size_stride(primals_154, (728, ), (1, ))
    assert_size_stride(primals_155, (728, ), (1, ))
    assert_size_stride(primals_156, (728, ), (1, ))
    assert_size_stride(primals_157, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_158, (728, ), (1, ))
    assert_size_stride(primals_159, (728, ), (1, ))
    assert_size_stride(primals_160, (728, ), (1, ))
    assert_size_stride(primals_161, (728, ), (1, ))
    assert_size_stride(primals_162, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_163, (728, ), (1, ))
    assert_size_stride(primals_164, (728, ), (1, ))
    assert_size_stride(primals_165, (728, ), (1, ))
    assert_size_stride(primals_166, (728, ), (1, ))
    assert_size_stride(primals_167, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_168, (728, ), (1, ))
    assert_size_stride(primals_169, (728, ), (1, ))
    assert_size_stride(primals_170, (728, ), (1, ))
    assert_size_stride(primals_171, (728, ), (1, ))
    assert_size_stride(primals_172, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_173, (728, ), (1, ))
    assert_size_stride(primals_174, (728, ), (1, ))
    assert_size_stride(primals_175, (728, ), (1, ))
    assert_size_stride(primals_176, (728, ), (1, ))
    assert_size_stride(primals_177, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_178, (728, ), (1, ))
    assert_size_stride(primals_179, (728, ), (1, ))
    assert_size_stride(primals_180, (728, ), (1, ))
    assert_size_stride(primals_181, (728, ), (1, ))
    assert_size_stride(primals_182, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_183, (728, ), (1, ))
    assert_size_stride(primals_184, (728, ), (1, ))
    assert_size_stride(primals_185, (728, ), (1, ))
    assert_size_stride(primals_186, (728, ), (1, ))
    assert_size_stride(primals_187, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_188, (728, ), (1, ))
    assert_size_stride(primals_189, (728, ), (1, ))
    assert_size_stride(primals_190, (728, ), (1, ))
    assert_size_stride(primals_191, (728, ), (1, ))
    assert_size_stride(primals_192, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_193, (728, ), (1, ))
    assert_size_stride(primals_194, (728, ), (1, ))
    assert_size_stride(primals_195, (728, ), (1, ))
    assert_size_stride(primals_196, (728, ), (1, ))
    assert_size_stride(primals_197, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_198, (728, ), (1, ))
    assert_size_stride(primals_199, (728, ), (1, ))
    assert_size_stride(primals_200, (728, ), (1, ))
    assert_size_stride(primals_201, (728, ), (1, ))
    assert_size_stride(primals_202, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_203, (728, ), (1, ))
    assert_size_stride(primals_204, (728, ), (1, ))
    assert_size_stride(primals_205, (728, ), (1, ))
    assert_size_stride(primals_206, (728, ), (1, ))
    assert_size_stride(primals_207, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_208, (728, ), (1, ))
    assert_size_stride(primals_209, (728, ), (1, ))
    assert_size_stride(primals_210, (728, ), (1, ))
    assert_size_stride(primals_211, (728, ), (1, ))
    assert_size_stride(primals_212, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_213, (728, ), (1, ))
    assert_size_stride(primals_214, (728, ), (1, ))
    assert_size_stride(primals_215, (728, ), (1, ))
    assert_size_stride(primals_216, (728, ), (1, ))
    assert_size_stride(primals_217, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_218, (728, ), (1, ))
    assert_size_stride(primals_219, (728, ), (1, ))
    assert_size_stride(primals_220, (728, ), (1, ))
    assert_size_stride(primals_221, (728, ), (1, ))
    assert_size_stride(primals_222, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_223, (728, ), (1, ))
    assert_size_stride(primals_224, (728, ), (1, ))
    assert_size_stride(primals_225, (728, ), (1, ))
    assert_size_stride(primals_226, (728, ), (1, ))
    assert_size_stride(primals_227, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_228, (728, ), (1, ))
    assert_size_stride(primals_229, (728, ), (1, ))
    assert_size_stride(primals_230, (728, ), (1, ))
    assert_size_stride(primals_231, (728, ), (1, ))
    assert_size_stride(primals_232, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_233, (728, ), (1, ))
    assert_size_stride(primals_234, (728, ), (1, ))
    assert_size_stride(primals_235, (728, ), (1, ))
    assert_size_stride(primals_236, (728, ), (1, ))
    assert_size_stride(primals_237, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_238, (728, ), (1, ))
    assert_size_stride(primals_239, (728, ), (1, ))
    assert_size_stride(primals_240, (728, ), (1, ))
    assert_size_stride(primals_241, (728, ), (1, ))
    assert_size_stride(primals_242, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_243, (728, ), (1, ))
    assert_size_stride(primals_244, (728, ), (1, ))
    assert_size_stride(primals_245, (728, ), (1, ))
    assert_size_stride(primals_246, (728, ), (1, ))
    assert_size_stride(primals_247, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_248, (728, ), (1, ))
    assert_size_stride(primals_249, (728, ), (1, ))
    assert_size_stride(primals_250, (728, ), (1, ))
    assert_size_stride(primals_251, (728, ), (1, ))
    assert_size_stride(primals_252, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_253, (728, ), (1, ))
    assert_size_stride(primals_254, (728, ), (1, ))
    assert_size_stride(primals_255, (728, ), (1, ))
    assert_size_stride(primals_256, (728, ), (1, ))
    assert_size_stride(primals_257, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_258, (728, ), (1, ))
    assert_size_stride(primals_259, (728, ), (1, ))
    assert_size_stride(primals_260, (728, ), (1, ))
    assert_size_stride(primals_261, (728, ), (1, ))
    assert_size_stride(primals_262, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_263, (728, ), (1, ))
    assert_size_stride(primals_264, (728, ), (1, ))
    assert_size_stride(primals_265, (728, ), (1, ))
    assert_size_stride(primals_266, (728, ), (1, ))
    assert_size_stride(primals_267, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_268, (728, ), (1, ))
    assert_size_stride(primals_269, (728, ), (1, ))
    assert_size_stride(primals_270, (728, ), (1, ))
    assert_size_stride(primals_271, (728, ), (1, ))
    assert_size_stride(primals_272, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_273, (728, ), (1, ))
    assert_size_stride(primals_274, (728, ), (1, ))
    assert_size_stride(primals_275, (728, ), (1, ))
    assert_size_stride(primals_276, (728, ), (1, ))
    assert_size_stride(primals_277, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_278, (728, ), (1, ))
    assert_size_stride(primals_279, (728, ), (1, ))
    assert_size_stride(primals_280, (728, ), (1, ))
    assert_size_stride(primals_281, (728, ), (1, ))
    assert_size_stride(primals_282, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_283, (728, ), (1, ))
    assert_size_stride(primals_284, (728, ), (1, ))
    assert_size_stride(primals_285, (728, ), (1, ))
    assert_size_stride(primals_286, (728, ), (1, ))
    assert_size_stride(primals_287, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_288, (728, ), (1, ))
    assert_size_stride(primals_289, (728, ), (1, ))
    assert_size_stride(primals_290, (728, ), (1, ))
    assert_size_stride(primals_291, (728, ), (1, ))
    assert_size_stride(primals_292, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_293, (728, ), (1, ))
    assert_size_stride(primals_294, (728, ), (1, ))
    assert_size_stride(primals_295, (728, ), (1, ))
    assert_size_stride(primals_296, (728, ), (1, ))
    assert_size_stride(primals_297, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_298, (728, ), (1, ))
    assert_size_stride(primals_299, (728, ), (1, ))
    assert_size_stride(primals_300, (728, ), (1, ))
    assert_size_stride(primals_301, (728, ), (1, ))
    assert_size_stride(primals_302, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_303, (728, ), (1, ))
    assert_size_stride(primals_304, (728, ), (1, ))
    assert_size_stride(primals_305, (728, ), (1, ))
    assert_size_stride(primals_306, (728, ), (1, ))
    assert_size_stride(primals_307, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_308, (728, ), (1, ))
    assert_size_stride(primals_309, (728, ), (1, ))
    assert_size_stride(primals_310, (728, ), (1, ))
    assert_size_stride(primals_311, (728, ), (1, ))
    assert_size_stride(primals_312, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_313, (728, ), (1, ))
    assert_size_stride(primals_314, (728, ), (1, ))
    assert_size_stride(primals_315, (728, ), (1, ))
    assert_size_stride(primals_316, (728, ), (1, ))
    assert_size_stride(primals_317, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_318, (728, ), (1, ))
    assert_size_stride(primals_319, (728, ), (1, ))
    assert_size_stride(primals_320, (728, ), (1, ))
    assert_size_stride(primals_321, (728, ), (1, ))
    assert_size_stride(primals_322, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_323, (728, ), (1, ))
    assert_size_stride(primals_324, (728, ), (1, ))
    assert_size_stride(primals_325, (728, ), (1, ))
    assert_size_stride(primals_326, (728, ), (1, ))
    assert_size_stride(primals_327, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_328, (728, ), (1, ))
    assert_size_stride(primals_329, (728, ), (1, ))
    assert_size_stride(primals_330, (728, ), (1, ))
    assert_size_stride(primals_331, (728, ), (1, ))
    assert_size_stride(primals_332, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_333, (728, ), (1, ))
    assert_size_stride(primals_334, (728, ), (1, ))
    assert_size_stride(primals_335, (728, ), (1, ))
    assert_size_stride(primals_336, (728, ), (1, ))
    assert_size_stride(primals_337, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_338, (728, ), (1, ))
    assert_size_stride(primals_339, (728, ), (1, ))
    assert_size_stride(primals_340, (728, ), (1, ))
    assert_size_stride(primals_341, (728, ), (1, ))
    assert_size_stride(primals_342, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_343, (728, ), (1, ))
    assert_size_stride(primals_344, (728, ), (1, ))
    assert_size_stride(primals_345, (728, ), (1, ))
    assert_size_stride(primals_346, (728, ), (1, ))
    assert_size_stride(primals_347, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_348, (728, ), (1, ))
    assert_size_stride(primals_349, (728, ), (1, ))
    assert_size_stride(primals_350, (728, ), (1, ))
    assert_size_stride(primals_351, (728, ), (1, ))
    assert_size_stride(primals_352, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_353, (728, ), (1, ))
    assert_size_stride(primals_354, (728, ), (1, ))
    assert_size_stride(primals_355, (728, ), (1, ))
    assert_size_stride(primals_356, (728, ), (1, ))
    assert_size_stride(primals_357, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_358, (728, ), (1, ))
    assert_size_stride(primals_359, (728, ), (1, ))
    assert_size_stride(primals_360, (728, ), (1, ))
    assert_size_stride(primals_361, (728, ), (1, ))
    assert_size_stride(primals_362, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_363, (728, ), (1, ))
    assert_size_stride(primals_364, (728, ), (1, ))
    assert_size_stride(primals_365, (728, ), (1, ))
    assert_size_stride(primals_366, (728, ), (1, ))
    assert_size_stride(primals_367, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_368, (728, ), (1, ))
    assert_size_stride(primals_369, (728, ), (1, ))
    assert_size_stride(primals_370, (728, ), (1, ))
    assert_size_stride(primals_371, (728, ), (1, ))
    assert_size_stride(primals_372, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_373, (728, ), (1, ))
    assert_size_stride(primals_374, (728, ), (1, ))
    assert_size_stride(primals_375, (728, ), (1, ))
    assert_size_stride(primals_376, (728, ), (1, ))
    assert_size_stride(primals_377, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_378, (728, ), (1, ))
    assert_size_stride(primals_379, (728, ), (1, ))
    assert_size_stride(primals_380, (728, ), (1, ))
    assert_size_stride(primals_381, (728, ), (1, ))
    assert_size_stride(primals_382, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_383, (728, ), (1, ))
    assert_size_stride(primals_384, (728, ), (1, ))
    assert_size_stride(primals_385, (728, ), (1, ))
    assert_size_stride(primals_386, (728, ), (1, ))
    assert_size_stride(primals_387, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_388, (728, ), (1, ))
    assert_size_stride(primals_389, (728, ), (1, ))
    assert_size_stride(primals_390, (728, ), (1, ))
    assert_size_stride(primals_391, (728, ), (1, ))
    assert_size_stride(primals_392, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_393, (728, ), (1, ))
    assert_size_stride(primals_394, (728, ), (1, ))
    assert_size_stride(primals_395, (728, ), (1, ))
    assert_size_stride(primals_396, (728, ), (1, ))
    assert_size_stride(primals_397, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_398, (728, ), (1, ))
    assert_size_stride(primals_399, (728, ), (1, ))
    assert_size_stride(primals_400, (728, ), (1, ))
    assert_size_stride(primals_401, (728, ), (1, ))
    assert_size_stride(primals_402, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_403, (728, ), (1, ))
    assert_size_stride(primals_404, (728, ), (1, ))
    assert_size_stride(primals_405, (728, ), (1, ))
    assert_size_stride(primals_406, (728, ), (1, ))
    assert_size_stride(primals_407, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_408, (728, ), (1, ))
    assert_size_stride(primals_409, (728, ), (1, ))
    assert_size_stride(primals_410, (728, ), (1, ))
    assert_size_stride(primals_411, (728, ), (1, ))
    assert_size_stride(primals_412, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_413, (728, ), (1, ))
    assert_size_stride(primals_414, (728, ), (1, ))
    assert_size_stride(primals_415, (728, ), (1, ))
    assert_size_stride(primals_416, (728, ), (1, ))
    assert_size_stride(primals_417, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_418, (728, ), (1, ))
    assert_size_stride(primals_419, (728, ), (1, ))
    assert_size_stride(primals_420, (728, ), (1, ))
    assert_size_stride(primals_421, (728, ), (1, ))
    assert_size_stride(primals_422, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_423, (728, ), (1, ))
    assert_size_stride(primals_424, (728, ), (1, ))
    assert_size_stride(primals_425, (728, ), (1, ))
    assert_size_stride(primals_426, (728, ), (1, ))
    assert_size_stride(primals_427, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_428, (728, ), (1, ))
    assert_size_stride(primals_429, (728, ), (1, ))
    assert_size_stride(primals_430, (728, ), (1, ))
    assert_size_stride(primals_431, (728, ), (1, ))
    assert_size_stride(primals_432, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_433, (728, ), (1, ))
    assert_size_stride(primals_434, (728, ), (1, ))
    assert_size_stride(primals_435, (728, ), (1, ))
    assert_size_stride(primals_436, (728, ), (1, ))
    assert_size_stride(primals_437, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_438, (728, ), (1, ))
    assert_size_stride(primals_439, (728, ), (1, ))
    assert_size_stride(primals_440, (728, ), (1, ))
    assert_size_stride(primals_441, (728, ), (1, ))
    assert_size_stride(primals_442, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_443, (728, ), (1, ))
    assert_size_stride(primals_444, (728, ), (1, ))
    assert_size_stride(primals_445, (728, ), (1, ))
    assert_size_stride(primals_446, (728, ), (1, ))
    assert_size_stride(primals_447, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_448, (728, ), (1, ))
    assert_size_stride(primals_449, (728, ), (1, ))
    assert_size_stride(primals_450, (728, ), (1, ))
    assert_size_stride(primals_451, (728, ), (1, ))
    assert_size_stride(primals_452, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_453, (728, ), (1, ))
    assert_size_stride(primals_454, (728, ), (1, ))
    assert_size_stride(primals_455, (728, ), (1, ))
    assert_size_stride(primals_456, (728, ), (1, ))
    assert_size_stride(primals_457, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_458, (728, ), (1, ))
    assert_size_stride(primals_459, (728, ), (1, ))
    assert_size_stride(primals_460, (728, ), (1, ))
    assert_size_stride(primals_461, (728, ), (1, ))
    assert_size_stride(primals_462, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_463, (728, ), (1, ))
    assert_size_stride(primals_464, (728, ), (1, ))
    assert_size_stride(primals_465, (728, ), (1, ))
    assert_size_stride(primals_466, (728, ), (1, ))
    assert_size_stride(primals_467, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_468, (728, ), (1, ))
    assert_size_stride(primals_469, (728, ), (1, ))
    assert_size_stride(primals_470, (728, ), (1, ))
    assert_size_stride(primals_471, (728, ), (1, ))
    assert_size_stride(primals_472, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_473, (728, ), (1, ))
    assert_size_stride(primals_474, (728, ), (1, ))
    assert_size_stride(primals_475, (728, ), (1, ))
    assert_size_stride(primals_476, (728, ), (1, ))
    assert_size_stride(primals_477, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_478, (728, ), (1, ))
    assert_size_stride(primals_479, (728, ), (1, ))
    assert_size_stride(primals_480, (728, ), (1, ))
    assert_size_stride(primals_481, (728, ), (1, ))
    assert_size_stride(primals_482, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_483, (728, ), (1, ))
    assert_size_stride(primals_484, (728, ), (1, ))
    assert_size_stride(primals_485, (728, ), (1, ))
    assert_size_stride(primals_486, (728, ), (1, ))
    assert_size_stride(primals_487, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_488, (728, ), (1, ))
    assert_size_stride(primals_489, (728, ), (1, ))
    assert_size_stride(primals_490, (728, ), (1, ))
    assert_size_stride(primals_491, (728, ), (1, ))
    assert_size_stride(primals_492, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_493, (728, ), (1, ))
    assert_size_stride(primals_494, (728, ), (1, ))
    assert_size_stride(primals_495, (728, ), (1, ))
    assert_size_stride(primals_496, (728, ), (1, ))
    assert_size_stride(primals_497, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_498, (728, ), (1, ))
    assert_size_stride(primals_499, (728, ), (1, ))
    assert_size_stride(primals_500, (728, ), (1, ))
    assert_size_stride(primals_501, (728, ), (1, ))
    assert_size_stride(primals_502, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_503, (728, ), (1, ))
    assert_size_stride(primals_504, (728, ), (1, ))
    assert_size_stride(primals_505, (728, ), (1, ))
    assert_size_stride(primals_506, (728, ), (1, ))
    assert_size_stride(primals_507, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_508, (728, ), (1, ))
    assert_size_stride(primals_509, (728, ), (1, ))
    assert_size_stride(primals_510, (728, ), (1, ))
    assert_size_stride(primals_511, (728, ), (1, ))
    assert_size_stride(primals_512, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_513, (728, ), (1, ))
    assert_size_stride(primals_514, (728, ), (1, ))
    assert_size_stride(primals_515, (728, ), (1, ))
    assert_size_stride(primals_516, (728, ), (1, ))
    assert_size_stride(primals_517, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_518, (728, ), (1, ))
    assert_size_stride(primals_519, (728, ), (1, ))
    assert_size_stride(primals_520, (728, ), (1, ))
    assert_size_stride(primals_521, (728, ), (1, ))
    assert_size_stride(primals_522, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_523, (728, ), (1, ))
    assert_size_stride(primals_524, (728, ), (1, ))
    assert_size_stride(primals_525, (728, ), (1, ))
    assert_size_stride(primals_526, (728, ), (1, ))
    assert_size_stride(primals_527, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_528, (728, ), (1, ))
    assert_size_stride(primals_529, (728, ), (1, ))
    assert_size_stride(primals_530, (728, ), (1, ))
    assert_size_stride(primals_531, (728, ), (1, ))
    assert_size_stride(primals_532, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_533, (728, ), (1, ))
    assert_size_stride(primals_534, (728, ), (1, ))
    assert_size_stride(primals_535, (728, ), (1, ))
    assert_size_stride(primals_536, (728, ), (1, ))
    assert_size_stride(primals_537, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_538, (728, ), (1, ))
    assert_size_stride(primals_539, (728, ), (1, ))
    assert_size_stride(primals_540, (728, ), (1, ))
    assert_size_stride(primals_541, (728, ), (1, ))
    assert_size_stride(primals_542, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_543, (728, ), (1, ))
    assert_size_stride(primals_544, (728, ), (1, ))
    assert_size_stride(primals_545, (728, ), (1, ))
    assert_size_stride(primals_546, (728, ), (1, ))
    assert_size_stride(primals_547, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_548, (728, ), (1, ))
    assert_size_stride(primals_549, (728, ), (1, ))
    assert_size_stride(primals_550, (728, ), (1, ))
    assert_size_stride(primals_551, (728, ), (1, ))
    assert_size_stride(primals_552, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_553, (728, ), (1, ))
    assert_size_stride(primals_554, (728, ), (1, ))
    assert_size_stride(primals_555, (728, ), (1, ))
    assert_size_stride(primals_556, (728, ), (1, ))
    assert_size_stride(primals_557, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_558, (728, ), (1, ))
    assert_size_stride(primals_559, (728, ), (1, ))
    assert_size_stride(primals_560, (728, ), (1, ))
    assert_size_stride(primals_561, (728, ), (1, ))
    assert_size_stride(primals_562, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_563, (728, ), (1, ))
    assert_size_stride(primals_564, (728, ), (1, ))
    assert_size_stride(primals_565, (728, ), (1, ))
    assert_size_stride(primals_566, (728, ), (1, ))
    assert_size_stride(primals_567, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_568, (728, ), (1, ))
    assert_size_stride(primals_569, (728, ), (1, ))
    assert_size_stride(primals_570, (728, ), (1, ))
    assert_size_stride(primals_571, (728, ), (1, ))
    assert_size_stride(primals_572, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_573, (728, ), (1, ))
    assert_size_stride(primals_574, (728, ), (1, ))
    assert_size_stride(primals_575, (728, ), (1, ))
    assert_size_stride(primals_576, (728, ), (1, ))
    assert_size_stride(primals_577, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_578, (728, ), (1, ))
    assert_size_stride(primals_579, (728, ), (1, ))
    assert_size_stride(primals_580, (728, ), (1, ))
    assert_size_stride(primals_581, (728, ), (1, ))
    assert_size_stride(primals_582, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_583, (728, ), (1, ))
    assert_size_stride(primals_584, (728, ), (1, ))
    assert_size_stride(primals_585, (728, ), (1, ))
    assert_size_stride(primals_586, (728, ), (1, ))
    assert_size_stride(primals_587, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_588, (728, ), (1, ))
    assert_size_stride(primals_589, (728, ), (1, ))
    assert_size_stride(primals_590, (728, ), (1, ))
    assert_size_stride(primals_591, (728, ), (1, ))
    assert_size_stride(primals_592, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_593, (728, ), (1, ))
    assert_size_stride(primals_594, (728, ), (1, ))
    assert_size_stride(primals_595, (728, ), (1, ))
    assert_size_stride(primals_596, (728, ), (1, ))
    assert_size_stride(primals_597, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_598, (728, ), (1, ))
    assert_size_stride(primals_599, (728, ), (1, ))
    assert_size_stride(primals_600, (728, ), (1, ))
    assert_size_stride(primals_601, (728, ), (1, ))
    assert_size_stride(primals_602, (728, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_603, (728, ), (1, ))
    assert_size_stride(primals_604, (728, ), (1, ))
    assert_size_stride(primals_605, (728, ), (1, ))
    assert_size_stride(primals_606, (728, ), (1, ))
    assert_size_stride(primals_607, (728, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_608, (728, ), (1, ))
    assert_size_stride(primals_609, (728, ), (1, ))
    assert_size_stride(primals_610, (728, ), (1, ))
    assert_size_stride(primals_611, (728, ), (1, ))
    assert_size_stride(primals_612, (1024, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_613, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_614, (1024, ), (1, ))
    assert_size_stride(primals_615, (1024, ), (1, ))
    assert_size_stride(primals_616, (1024, ), (1, ))
    assert_size_stride(primals_617, (1024, ), (1, ))
    assert_size_stride(primals_618, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_619, (1024, ), (1, ))
    assert_size_stride(primals_620, (1024, ), (1, ))
    assert_size_stride(primals_621, (1024, ), (1, ))
    assert_size_stride(primals_622, (1024, ), (1, ))
    assert_size_stride(primals_623, (1024, 728, 1, 1), (728, 1, 1, 1))
    assert_size_stride(primals_624, (1024, ), (1, ))
    assert_size_stride(primals_625, (1024, ), (1, ))
    assert_size_stride(primals_626, (1024, ), (1, ))
    assert_size_stride(primals_627, (1024, ), (1, ))
    assert_size_stride(primals_628, (1024, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_629, (1024, ), (1, ))
    assert_size_stride(primals_630, (1024, ), (1, ))
    assert_size_stride(primals_631, (1024, ), (1, ))
    assert_size_stride(primals_632, (1024, ), (1, ))
    assert_size_stride(primals_633, (1536, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_634, (1536, ), (1, ))
    assert_size_stride(primals_635, (1536, ), (1, ))
    assert_size_stride(primals_636, (1536, ), (1, ))
    assert_size_stride(primals_637, (1536, ), (1, ))
    assert_size_stride(primals_638, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_639, (1536, ), (1, ))
    assert_size_stride(primals_640, (1536, ), (1, ))
    assert_size_stride(primals_641, (1536, ), (1, ))
    assert_size_stride(primals_642, (1536, ), (1, ))
    assert_size_stride(primals_643, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_644, (1536, ), (1, ))
    assert_size_stride(primals_645, (1536, ), (1, ))
    assert_size_stride(primals_646, (1536, ), (1, ))
    assert_size_stride(primals_647, (1536, ), (1, ))
    assert_size_stride(primals_648, (1536, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_649, (1536, ), (1, ))
    assert_size_stride(primals_650, (1536, ), (1, ))
    assert_size_stride(primals_651, (1536, ), (1, ))
    assert_size_stride(primals_652, (1536, ), (1, ))
    assert_size_stride(primals_653, (2048, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_654, (2048, ), (1, ))
    assert_size_stride(primals_655, (2048, ), (1, ))
    assert_size_stride(primals_656, (2048, ), (1, ))
    assert_size_stride(primals_657, (2048, ), (1, ))
    assert_size_stride(primals_658, (1000, 2048), (2048, 1))
    assert_size_stride(primals_659, (1000, ), (1, ))
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
        buf3 = extern_kernels.convolution(buf1, buf0, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (4, 32, 32, 32), (32768, 1, 1024, 32))
        buf4 = empty_strided_cuda((4, 32, 32, 32), (32768, 1, 1024, 32), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf3, primals_3, primals_4, primals_5, primals_6, buf4, 131072, grid=grid(131072), stream=stream0)
        del primals_6
        # Topologically Sorted Source Nodes: [x_3], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf6 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf5, primals_8, primals_9, primals_10, primals_11, buf6, 262144, grid=grid(262144), stream=stream0)
        del primals_11
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf7, (4, 64, 32, 32), (65536, 1, 2048, 64))
        buf8 = empty_strided_cuda((4, 64, 32, 32), (65536, 1, 2048, 64), torch.float32)
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_5.run(buf7, primals_13, primals_14, primals_15, primals_16, buf8, 262144, grid=grid(262144), stream=stream0)
        del primals_16
        # Topologically Sorted Source Nodes: [x_8], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf10 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_1, input_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf9, primals_18, primals_19, primals_20, primals_21, buf10, 524288, grid=grid(524288), stream=stream0)
        del primals_21
        # Topologically Sorted Source Nodes: [x_9], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_22, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf11, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf12 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf11, primals_23, primals_24, primals_25, primals_26, buf12, 524288, grid=grid(524288), stream=stream0)
        del primals_26
        # Topologically Sorted Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 128, 32, 32), (131072, 1, 4096, 128))
        buf14 = empty_strided_cuda((4, 128, 32, 32), (131072, 1, 4096, 128), torch.float32)
        # Topologically Sorted Source Nodes: [input_3, input_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf13, primals_28, primals_29, primals_30, primals_31, buf14, 524288, grid=grid(524288), stream=stream0)
        del primals_31
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, primals_32, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf15, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf16 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf15, primals_33, primals_34, primals_35, primals_36, buf16, 131072, grid=grid(131072), stream=stream0)
        del primals_36
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 128, 16, 16), (32768, 1, 2048, 128))
        # Topologically Sorted Source Nodes: [conv2d_8], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf6, primals_42, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf19 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        buf20 = buf19; del buf19  # reuse
        # Topologically Sorted Source Nodes: [input_5, skip, out, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_9.run(buf20, buf17, primals_38, primals_39, primals_40, primals_41, buf18, primals_43, primals_44, primals_45, primals_46, 131072, grid=grid(131072), stream=stream0)
        del primals_41
        del primals_46
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_47, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf21, (4, 128, 16, 16), (32768, 1, 2048, 128))
        buf22 = empty_strided_cuda((4, 128, 16, 16), (32768, 1, 2048, 128), torch.float32)
        # Topologically Sorted Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_8.run(buf21, primals_48, primals_49, primals_50, primals_51, buf22, 131072, grid=grid(131072), stream=stream0)
        del primals_51
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf24 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_6, input_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf23, primals_53, primals_54, primals_55, primals_56, buf24, 262144, grid=grid(262144), stream=stream0)
        del primals_56
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, primals_57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf25, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf26 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_11.run(buf25, primals_58, primals_59, primals_60, primals_61, buf26, 262144, grid=grid(262144), stream=stream0)
        del primals_61
        # Topologically Sorted Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_62, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 256, 16, 16), (65536, 1, 4096, 256))
        buf28 = empty_strided_cuda((4, 256, 16, 16), (65536, 1, 4096, 256), torch.float32)
        # Topologically Sorted Source Nodes: [input_8, input_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf27, primals_63, primals_64, primals_65, primals_66, buf28, 262144, grid=grid(262144), stream=stream0)
        del primals_66
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, primals_67, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf29, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf30 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf29, primals_68, primals_69, primals_70, primals_71, buf30, 65536, grid=grid(65536), stream=stream0)
        del primals_71
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 256, 8, 8), (16384, 1, 2048, 256))
        # Topologically Sorted Source Nodes: [conv2d_15], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf20, primals_77, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf33 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        buf34 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [input_10, skip_1, out_1, input_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf34, buf31, primals_73, primals_74, primals_75, primals_76, buf32, primals_78, primals_79, primals_80, primals_81, 65536, grid=grid(65536), stream=stream0)
        del primals_76
        del primals_81
        # Topologically Sorted Source Nodes: [x_25], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf35, (4, 256, 8, 8), (16384, 1, 2048, 256))
        buf36 = empty_strided_cuda((4, 256, 8, 8), (16384, 1, 2048, 256), torch.float32)
        # Topologically Sorted Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_12.run(buf35, primals_83, primals_84, primals_85, primals_86, buf36, 65536, grid=grid(65536), stream=stream0)
        del primals_86
        # Topologically Sorted Source Nodes: [x_27], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 728, 8, 8), (46592, 1, 5824, 728))
        buf38 = empty_strided_cuda((4, 728, 8, 8), (46592, 1, 5824, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_12, input_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf37, primals_88, primals_89, primals_90, primals_91, buf38, 186368, grid=grid(186368), stream=stream0)
        del primals_91
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf39, (4, 728, 8, 8), (46592, 1, 5824, 728))
        buf40 = empty_strided_cuda((4, 728, 8, 8), (46592, 1, 5824, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf39, primals_93, primals_94, primals_95, primals_96, buf40, 186368, grid=grid(186368), stream=stream0)
        del primals_96
        # Topologically Sorted Source Nodes: [x_30], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 728, 8, 8), (46592, 1, 5824, 728))
        buf42 = empty_strided_cuda((4, 728, 8, 8), (46592, 1, 5824, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_14, input_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf41, primals_98, primals_99, primals_100, primals_101, buf42, 186368, grid=grid(186368), stream=stream0)
        del primals_101
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_102, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf43, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf44 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf43, primals_103, primals_104, primals_105, primals_106, buf44, 46592, grid=grid(46592), stream=stream0)
        del primals_106
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_107, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 728, 4, 4), (11648, 1, 2912, 728))
        # Topologically Sorted Source Nodes: [conv2d_22], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf34, primals_112, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf47 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        buf48 = buf47; del buf47  # reuse
        # Topologically Sorted Source Nodes: [input_16, skip_2, out_2, input_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf48, buf45, primals_108, primals_109, primals_110, primals_111, buf46, primals_113, primals_114, primals_115, primals_116, 46592, grid=grid(46592), stream=stream0)
        del primals_111
        del primals_116
        # Topologically Sorted Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_117, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf49, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf50 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf49, primals_118, primals_119, primals_120, primals_121, buf50, 46592, grid=grid(46592), stream=stream0)
        del primals_121
        # Topologically Sorted Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, primals_122, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf52 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_18, input_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf51, primals_123, primals_124, primals_125, primals_126, buf52, 46592, grid=grid(46592), stream=stream0)
        del primals_126
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_127, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf53, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf54 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf53, primals_128, primals_129, primals_130, primals_131, buf54, 46592, grid=grid(46592), stream=stream0)
        del primals_131
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf56 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_20, input_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf55, primals_133, primals_134, primals_135, primals_136, buf56, 46592, grid=grid(46592), stream=stream0)
        del primals_136
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_137, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf57, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf58 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf57, primals_138, primals_139, primals_140, primals_141, buf58, 46592, grid=grid(46592), stream=stream0)
        del primals_141
        # Topologically Sorted Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf60 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_22, out_3, input_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf59, primals_143, primals_144, primals_145, primals_146, buf48, buf60, 46592, grid=grid(46592), stream=stream0)
        del primals_146
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf61, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf62 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_44], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf61, primals_148, primals_149, primals_150, primals_151, buf62, 46592, grid=grid(46592), stream=stream0)
        del primals_151
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, primals_152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf64 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_24, input_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf63, primals_153, primals_154, primals_155, primals_156, buf64, 46592, grid=grid(46592), stream=stream0)
        del primals_156
        # Topologically Sorted Source Nodes: [x_46], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf65, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf66 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_47], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf65, primals_158, primals_159, primals_160, primals_161, buf66, 46592, grid=grid(46592), stream=stream0)
        del primals_161
        # Topologically Sorted Source Nodes: [x_48], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf68 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_26, input_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf67, primals_163, primals_164, primals_165, primals_166, buf68, 46592, grid=grid(46592), stream=stream0)
        del primals_166
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf69, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf70 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf69, primals_168, primals_169, primals_170, primals_171, buf70, 46592, grid=grid(46592), stream=stream0)
        del primals_171
        # Topologically Sorted Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf72 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_28, out_4, input_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf71, primals_173, primals_174, primals_175, primals_176, buf60, buf72, 46592, grid=grid(46592), stream=stream0)
        del primals_176
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf73, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf74 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf73, primals_178, primals_179, primals_180, primals_181, buf74, 46592, grid=grid(46592), stream=stream0)
        del primals_181
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf76 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_30, input_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf75, primals_183, primals_184, primals_185, primals_186, buf76, 46592, grid=grid(46592), stream=stream0)
        del primals_186
        # Topologically Sorted Source Nodes: [x_55], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf77, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf78 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf77, primals_188, primals_189, primals_190, primals_191, buf78, 46592, grid=grid(46592), stream=stream0)
        del primals_191
        # Topologically Sorted Source Nodes: [x_57], Original ATen: [aten.convolution]
        buf79 = extern_kernels.convolution(buf78, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf80 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_32, input_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf79, primals_193, primals_194, primals_195, primals_196, buf80, 46592, grid=grid(46592), stream=stream0)
        del primals_196
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, primals_197, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf81, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf82 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf81, primals_198, primals_199, primals_200, primals_201, buf82, 46592, grid=grid(46592), stream=stream0)
        del primals_201
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf84 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_34, out_5, input_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf83, primals_203, primals_204, primals_205, primals_206, buf72, buf84, 46592, grid=grid(46592), stream=stream0)
        del primals_206
        # Topologically Sorted Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf85, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf86 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf85, primals_208, primals_209, primals_210, primals_211, buf86, 46592, grid=grid(46592), stream=stream0)
        del primals_211
        # Topologically Sorted Source Nodes: [x_63], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf88 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_36, input_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf87, primals_213, primals_214, primals_215, primals_216, buf88, 46592, grid=grid(46592), stream=stream0)
        del primals_216
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf89, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf90 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_65], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf89, primals_218, primals_219, primals_220, primals_221, buf90, 46592, grid=grid(46592), stream=stream0)
        del primals_221
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf92 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_38, input_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf91, primals_223, primals_224, primals_225, primals_226, buf92, 46592, grid=grid(46592), stream=stream0)
        del primals_226
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf93, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf94 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf93, primals_228, primals_229, primals_230, primals_231, buf94, 46592, grid=grid(46592), stream=stream0)
        del primals_231
        # Topologically Sorted Source Nodes: [x_69], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_232, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf96 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_40, out_6, input_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf95, primals_233, primals_234, primals_235, primals_236, buf84, buf96, 46592, grid=grid(46592), stream=stream0)
        del primals_236
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf97, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf98 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf97, primals_238, primals_239, primals_240, primals_241, buf98, 46592, grid=grid(46592), stream=stream0)
        del primals_241
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, primals_242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf100 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_42, input_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf99, primals_243, primals_244, primals_245, primals_246, buf100, 46592, grid=grid(46592), stream=stream0)
        del primals_246
        # Topologically Sorted Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf101, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf102 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf101, primals_248, primals_249, primals_250, primals_251, buf102, 46592, grid=grid(46592), stream=stream0)
        del primals_251
        # Topologically Sorted Source Nodes: [x_75], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf104 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_44, input_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf103, primals_253, primals_254, primals_255, primals_256, buf104, 46592, grid=grid(46592), stream=stream0)
        del primals_256
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf105, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf106 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf105, primals_258, primals_259, primals_260, primals_261, buf106, 46592, grid=grid(46592), stream=stream0)
        del primals_261
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf108 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_46, out_7, input_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf107, primals_263, primals_264, primals_265, primals_266, buf96, buf108, 46592, grid=grid(46592), stream=stream0)
        del primals_266
        # Topologically Sorted Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf109, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf110 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf109, primals_268, primals_269, primals_270, primals_271, buf110, 46592, grid=grid(46592), stream=stream0)
        del primals_271
        # Topologically Sorted Source Nodes: [x_81], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf112 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_48, input_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf111, primals_273, primals_274, primals_275, primals_276, buf112, 46592, grid=grid(46592), stream=stream0)
        del primals_276
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf113, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf114 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf113, primals_278, primals_279, primals_280, primals_281, buf114, 46592, grid=grid(46592), stream=stream0)
        del primals_281
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, primals_282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf116 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_50, input_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf115, primals_283, primals_284, primals_285, primals_286, buf116, 46592, grid=grid(46592), stream=stream0)
        del primals_286
        # Topologically Sorted Source Nodes: [x_85], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, primals_287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf117, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf118 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf117, primals_288, primals_289, primals_290, primals_291, buf118, 46592, grid=grid(46592), stream=stream0)
        del primals_291
        # Topologically Sorted Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf120 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_52, out_8, input_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf119, primals_293, primals_294, primals_295, primals_296, buf108, buf120, 46592, grid=grid(46592), stream=stream0)
        del primals_296
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf121, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf122 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf121, primals_298, primals_299, primals_300, primals_301, buf122, 46592, grid=grid(46592), stream=stream0)
        del primals_301
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf124 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_54, input_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf123, primals_303, primals_304, primals_305, primals_306, buf124, 46592, grid=grid(46592), stream=stream0)
        del primals_306
        # Topologically Sorted Source Nodes: [x_91], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf125, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf126 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf125, primals_308, primals_309, primals_310, primals_311, buf126, 46592, grid=grid(46592), stream=stream0)
        del primals_311
        # Topologically Sorted Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_312, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf128 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_56, input_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf127, primals_313, primals_314, primals_315, primals_316, buf128, 46592, grid=grid(46592), stream=stream0)
        del primals_316
        # Topologically Sorted Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, primals_317, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf129, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf130 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf129, primals_318, primals_319, primals_320, primals_321, buf130, 46592, grid=grid(46592), stream=stream0)
        del primals_321
        # Topologically Sorted Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf132 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_58, out_9, input_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf131, primals_323, primals_324, primals_325, primals_326, buf120, buf132, 46592, grid=grid(46592), stream=stream0)
        del primals_326
        # Topologically Sorted Source Nodes: [x_97], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_327, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf133, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf134 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_98], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf133, primals_328, primals_329, primals_330, primals_331, buf134, 46592, grid=grid(46592), stream=stream0)
        del primals_331
        # Topologically Sorted Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf136 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_60, input_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf135, primals_333, primals_334, primals_335, primals_336, buf136, 46592, grid=grid(46592), stream=stream0)
        del primals_336
        # Topologically Sorted Source Nodes: [x_100], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf137, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf138 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf137, primals_338, primals_339, primals_340, primals_341, buf138, 46592, grid=grid(46592), stream=stream0)
        del primals_341
        # Topologically Sorted Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf139 = extern_kernels.convolution(buf138, primals_342, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf140 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_62, input_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf139, primals_343, primals_344, primals_345, primals_346, buf140, 46592, grid=grid(46592), stream=stream0)
        del primals_346
        # Topologically Sorted Source Nodes: [x_103], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_347, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf141, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf142 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf141, primals_348, primals_349, primals_350, primals_351, buf142, 46592, grid=grid(46592), stream=stream0)
        del primals_351
        # Topologically Sorted Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_352, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf144 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_64, out_10, input_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf143, primals_353, primals_354, primals_355, primals_356, buf132, buf144, 46592, grid=grid(46592), stream=stream0)
        del primals_356
        # Topologically Sorted Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf144, primals_357, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf145, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf146 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_107], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf145, primals_358, primals_359, primals_360, primals_361, buf146, 46592, grid=grid(46592), stream=stream0)
        del primals_361
        # Topologically Sorted Source Nodes: [x_108], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf148 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_66, input_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf147, primals_363, primals_364, primals_365, primals_366, buf148, 46592, grid=grid(46592), stream=stream0)
        del primals_366
        # Topologically Sorted Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf149, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf150 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf149, primals_368, primals_369, primals_370, primals_371, buf150, 46592, grid=grid(46592), stream=stream0)
        del primals_371
        # Topologically Sorted Source Nodes: [x_111], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf150, primals_372, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf152 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_68, input_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf151, primals_373, primals_374, primals_375, primals_376, buf152, 46592, grid=grid(46592), stream=stream0)
        del primals_376
        # Topologically Sorted Source Nodes: [x_112], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf153, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf154 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_113], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf153, primals_378, primals_379, primals_380, primals_381, buf154, 46592, grid=grid(46592), stream=stream0)
        del primals_381
        # Topologically Sorted Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_382, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf155, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf156 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_70, out_11, input_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf155, primals_383, primals_384, primals_385, primals_386, buf144, buf156, 46592, grid=grid(46592), stream=stream0)
        del primals_386
        # Topologically Sorted Source Nodes: [x_115], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_387, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf157, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf158 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf157, primals_388, primals_389, primals_390, primals_391, buf158, 46592, grid=grid(46592), stream=stream0)
        del primals_391
        # Topologically Sorted Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf160 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_72, input_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf159, primals_393, primals_394, primals_395, primals_396, buf160, 46592, grid=grid(46592), stream=stream0)
        del primals_396
        # Topologically Sorted Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf161, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf162 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf161, primals_398, primals_399, primals_400, primals_401, buf162, 46592, grid=grid(46592), stream=stream0)
        del primals_401
        # Topologically Sorted Source Nodes: [x_120], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf164 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_74, input_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf163, primals_403, primals_404, primals_405, primals_406, buf164, 46592, grid=grid(46592), stream=stream0)
        del primals_406
        # Topologically Sorted Source Nodes: [x_121], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_407, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf165, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf166 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_122], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf165, primals_408, primals_409, primals_410, primals_411, buf166, 46592, grid=grid(46592), stream=stream0)
        del primals_411
        # Topologically Sorted Source Nodes: [x_123], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, primals_412, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf168 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_76, out_12, input_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf167, primals_413, primals_414, primals_415, primals_416, buf156, buf168, 46592, grid=grid(46592), stream=stream0)
        del primals_416
        # Topologically Sorted Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_417, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf169, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf170 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf169, primals_418, primals_419, primals_420, primals_421, buf170, 46592, grid=grid(46592), stream=stream0)
        del primals_421
        # Topologically Sorted Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_422, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf172 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_78, input_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf171, primals_423, primals_424, primals_425, primals_426, buf172, 46592, grid=grid(46592), stream=stream0)
        del primals_426
        # Topologically Sorted Source Nodes: [x_127], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf173, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf174 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf173, primals_428, primals_429, primals_430, primals_431, buf174, 46592, grid=grid(46592), stream=stream0)
        del primals_431
        # Topologically Sorted Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_432, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf176 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_80, input_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf175, primals_433, primals_434, primals_435, primals_436, buf176, 46592, grid=grid(46592), stream=stream0)
        del primals_436
        # Topologically Sorted Source Nodes: [x_130], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_437, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf177, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf178 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_131], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf177, primals_438, primals_439, primals_440, primals_441, buf178, 46592, grid=grid(46592), stream=stream0)
        del primals_441
        # Topologically Sorted Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_442, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf180 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_82, out_13, input_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf179, primals_443, primals_444, primals_445, primals_446, buf168, buf180, 46592, grid=grid(46592), stream=stream0)
        del primals_446
        # Topologically Sorted Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf181 = extern_kernels.convolution(buf180, primals_447, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf181, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf182 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf181, primals_448, primals_449, primals_450, primals_451, buf182, 46592, grid=grid(46592), stream=stream0)
        del primals_451
        # Topologically Sorted Source Nodes: [x_135], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, primals_452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf184 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_84, input_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf183, primals_453, primals_454, primals_455, primals_456, buf184, 46592, grid=grid(46592), stream=stream0)
        del primals_456
        # Topologically Sorted Source Nodes: [x_136], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_457, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf185, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf186 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_137], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf185, primals_458, primals_459, primals_460, primals_461, buf186, 46592, grid=grid(46592), stream=stream0)
        del primals_461
        # Topologically Sorted Source Nodes: [x_138], Original ATen: [aten.convolution]
        buf187 = extern_kernels.convolution(buf186, primals_462, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf188 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_86, input_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf187, primals_463, primals_464, primals_465, primals_466, buf188, 46592, grid=grid(46592), stream=stream0)
        del primals_466
        # Topologically Sorted Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_467, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf189, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf190 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf189, primals_468, primals_469, primals_470, primals_471, buf190, 46592, grid=grid(46592), stream=stream0)
        del primals_471
        # Topologically Sorted Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_472, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf192 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_88, out_14, input_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf191, primals_473, primals_474, primals_475, primals_476, buf180, buf192, 46592, grid=grid(46592), stream=stream0)
        del primals_476
        # Topologically Sorted Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, primals_477, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf193, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf194 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf193, primals_478, primals_479, primals_480, primals_481, buf194, 46592, grid=grid(46592), stream=stream0)
        del primals_481
        # Topologically Sorted Source Nodes: [x_144], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, primals_482, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf196 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_90, input_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf195, primals_483, primals_484, primals_485, primals_486, buf196, 46592, grid=grid(46592), stream=stream0)
        del primals_486
        # Topologically Sorted Source Nodes: [x_145], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_487, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf197, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf198 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_146], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf197, primals_488, primals_489, primals_490, primals_491, buf198, 46592, grid=grid(46592), stream=stream0)
        del primals_491
        # Topologically Sorted Source Nodes: [x_147], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, primals_492, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf200 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_92, input_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf199, primals_493, primals_494, primals_495, primals_496, buf200, 46592, grid=grid(46592), stream=stream0)
        del primals_496
        # Topologically Sorted Source Nodes: [x_148], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_497, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf201, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf202 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_149], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf201, primals_498, primals_499, primals_500, primals_501, buf202, 46592, grid=grid(46592), stream=stream0)
        del primals_501
        # Topologically Sorted Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf204 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_94, out_15, input_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf203, primals_503, primals_504, primals_505, primals_506, buf192, buf204, 46592, grid=grid(46592), stream=stream0)
        del primals_506
        # Topologically Sorted Source Nodes: [x_151], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, primals_507, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf205, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf206 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_152], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf205, primals_508, primals_509, primals_510, primals_511, buf206, 46592, grid=grid(46592), stream=stream0)
        del primals_511
        # Topologically Sorted Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf208 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_96, input_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf207, primals_513, primals_514, primals_515, primals_516, buf208, 46592, grid=grid(46592), stream=stream0)
        del primals_516
        # Topologically Sorted Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_517, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf209, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf210 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf209, primals_518, primals_519, primals_520, primals_521, buf210, 46592, grid=grid(46592), stream=stream0)
        del primals_521
        # Topologically Sorted Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_522, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf212 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_98, input_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf211, primals_523, primals_524, primals_525, primals_526, buf212, 46592, grid=grid(46592), stream=stream0)
        del primals_526
        # Topologically Sorted Source Nodes: [x_157], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_527, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf213, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf214 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf213, primals_528, primals_529, primals_530, primals_531, buf214, 46592, grid=grid(46592), stream=stream0)
        del primals_531
        # Topologically Sorted Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_532, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf216 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_100, out_16, input_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf215, primals_533, primals_534, primals_535, primals_536, buf204, buf216, 46592, grid=grid(46592), stream=stream0)
        del primals_536
        # Topologically Sorted Source Nodes: [x_160], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_537, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf217, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf218 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf217, primals_538, primals_539, primals_540, primals_541, buf218, 46592, grid=grid(46592), stream=stream0)
        del primals_541
        # Topologically Sorted Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_542, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf220 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_102, input_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf219, primals_543, primals_544, primals_545, primals_546, buf220, 46592, grid=grid(46592), stream=stream0)
        del primals_546
        # Topologically Sorted Source Nodes: [x_163], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_547, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf221, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf222 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf221, primals_548, primals_549, primals_550, primals_551, buf222, 46592, grid=grid(46592), stream=stream0)
        del primals_551
        # Topologically Sorted Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_552, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf224 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_104, input_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf223, primals_553, primals_554, primals_555, primals_556, buf224, 46592, grid=grid(46592), stream=stream0)
        del primals_556
        # Topologically Sorted Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_557, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf225, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf226 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf225, primals_558, primals_559, primals_560, primals_561, buf226, 46592, grid=grid(46592), stream=stream0)
        del primals_561
        # Topologically Sorted Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_562, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf228 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_106, out_17, input_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf227, primals_563, primals_564, primals_565, primals_566, buf216, buf228, 46592, grid=grid(46592), stream=stream0)
        del primals_566
        # Topologically Sorted Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, primals_567, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf229, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf230 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_170], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf229, primals_568, primals_569, primals_570, primals_571, buf230, 46592, grid=grid(46592), stream=stream0)
        del primals_571
        # Topologically Sorted Source Nodes: [x_171], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(buf230, primals_572, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf232 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_108, input_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf231, primals_573, primals_574, primals_575, primals_576, buf232, 46592, grid=grid(46592), stream=stream0)
        del primals_576
        # Topologically Sorted Source Nodes: [x_172], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_577, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf233, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf234 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf233, primals_578, primals_579, primals_580, primals_581, buf234, 46592, grid=grid(46592), stream=stream0)
        del primals_581
        # Topologically Sorted Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf234, primals_582, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf236 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_110, input_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf235, primals_583, primals_584, primals_585, primals_586, buf236, 46592, grid=grid(46592), stream=stream0)
        del primals_586
        # Topologically Sorted Source Nodes: [x_175], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_587, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf237, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf238 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf237, primals_588, primals_589, primals_590, primals_591, buf238, 46592, grid=grid(46592), stream=stream0)
        del primals_591
        # Topologically Sorted Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_592, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf240 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_112, out_18, input_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_19.run(buf239, primals_593, primals_594, primals_595, primals_596, buf228, buf240, 46592, grid=grid(46592), stream=stream0)
        del primals_596
        # Topologically Sorted Source Nodes: [x_178], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_597, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf241, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf242 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf241, primals_598, primals_599, primals_600, primals_601, buf242, 46592, grid=grid(46592), stream=stream0)
        del primals_601
        # Topologically Sorted Source Nodes: [x_180], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf242, primals_602, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf244 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [input_114, input_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf243, primals_603, primals_604, primals_605, primals_606, buf244, 46592, grid=grid(46592), stream=stream0)
        del primals_606
        # Topologically Sorted Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_607, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=728, bias=None)
        assert_size_stride(buf245, (4, 728, 4, 4), (11648, 1, 2912, 728))
        buf246 = empty_strided_cuda((4, 728, 4, 4), (11648, 1, 2912, 728), torch.float32)
        # Topologically Sorted Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_16.run(buf245, primals_608, primals_609, primals_610, primals_611, buf246, 46592, grid=grid(46592), stream=stream0)
        del primals_611
        # Topologically Sorted Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_612, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (4, 1024, 4, 4), (16384, 1, 4096, 1024))
        buf248 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [input_116], Original ATen: [aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_relu_20.run(buf248, 65536, grid=grid(65536), stream=stream0)
        # Topologically Sorted Source Nodes: [x_184], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, primals_613, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf249, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf250 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf249, primals_614, primals_615, primals_616, primals_617, buf250, 16384, grid=grid(16384), stream=stream0)
        del primals_617
        # Topologically Sorted Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_618, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        # Topologically Sorted Source Nodes: [conv2d_125], Original ATen: [aten.convolution]
        buf252 = extern_kernels.convolution(buf240, primals_623, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf253 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        buf254 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [input_117, skip_3, out_19, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_22.run(buf254, buf251, primals_619, primals_620, primals_621, primals_622, buf252, primals_624, primals_625, primals_626, primals_627, 16384, grid=grid(16384), stream=stream0)
        del primals_622
        del primals_627
        # Topologically Sorted Source Nodes: [x_188], Original ATen: [aten.convolution]
        buf255 = extern_kernels.convolution(buf254, primals_628, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1024, bias=None)
        assert_size_stride(buf255, (4, 1024, 2, 2), (4096, 1, 2048, 1024))
        buf256 = empty_strided_cuda((4, 1024, 2, 2), (4096, 1, 2048, 1024), torch.float32)
        # Topologically Sorted Source Nodes: [x_189], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf255, primals_629, primals_630, primals_631, primals_632, buf256, 16384, grid=grid(16384), stream=stream0)
        del primals_632
        # Topologically Sorted Source Nodes: [x_190], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_633, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf258 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_191, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf257, primals_634, primals_635, primals_636, primals_637, buf258, 24576, grid=grid(24576), stream=stream0)
        del primals_637
        # Topologically Sorted Source Nodes: [x_193], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_638, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf259, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf260 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf259, primals_639, primals_640, primals_641, primals_642, buf260, 24576, grid=grid(24576), stream=stream0)
        del primals_642
        # Topologically Sorted Source Nodes: [x_195], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf260, primals_643, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf262 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_196, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf261, primals_644, primals_645, primals_646, primals_647, buf262, 24576, grid=grid(24576), stream=stream0)
        del primals_647
        # Topologically Sorted Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_648, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1536, bias=None)
        assert_size_stride(buf263, (4, 1536, 2, 2), (6144, 1, 3072, 1536))
        buf264 = empty_strided_cuda((4, 1536, 2, 2), (6144, 1, 3072, 1536), torch.float32)
        # Topologically Sorted Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_no_training]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_24.run(buf263, primals_649, primals_650, primals_651, primals_652, buf264, 24576, grid=grid(24576), stream=stream0)
        del primals_652
        # Topologically Sorted Source Nodes: [x_200], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf264, primals_653, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (4, 2048, 2, 2), (8192, 1, 4096, 2048))
        buf266 = empty_strided_cuda((4, 2048, 1, 1), (2048, 1, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_201, x_202, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.mean]
        stream0 = get_raw_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_mean_relu_25.run(buf265, primals_654, primals_655, primals_656, primals_657, buf266, 8192, grid=grid(8192), stream=stream0)
        buf267 = empty_strided_cuda((4, 1000), (1000, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_205], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_659, reinterpret_tensor(buf266, (4, 2048), (2048, 1), 0), reinterpret_tensor(primals_658, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf267)
        del primals_659
    return (buf267, buf0, buf1, primals_3, primals_4, primals_5, buf2, primals_8, primals_9, primals_10, primals_12, primals_13, primals_14, primals_15, primals_17, primals_18, primals_19, primals_20, primals_22, primals_23, primals_24, primals_25, primals_27, primals_28, primals_29, primals_30, primals_32, primals_33, primals_34, primals_35, primals_37, primals_38, primals_39, primals_40, primals_42, primals_43, primals_44, primals_45, primals_47, primals_48, primals_49, primals_50, primals_52, primals_53, primals_54, primals_55, primals_57, primals_58, primals_59, primals_60, primals_62, primals_63, primals_64, primals_65, primals_67, primals_68, primals_69, primals_70, primals_72, primals_73, primals_74, primals_75, primals_77, primals_78, primals_79, primals_80, primals_82, primals_83, primals_84, primals_85, primals_87, primals_88, primals_89, primals_90, primals_92, primals_93, primals_94, primals_95, primals_97, primals_98, primals_99, primals_100, primals_102, primals_103, primals_104, primals_105, primals_107, primals_108, primals_109, primals_110, primals_112, primals_113, primals_114, primals_115, primals_117, primals_118, primals_119, primals_120, primals_122, primals_123, primals_124, primals_125, primals_127, primals_128, primals_129, primals_130, primals_132, primals_133, primals_134, primals_135, primals_137, primals_138, primals_139, primals_140, primals_142, primals_143, primals_144, primals_145, primals_147, primals_148, primals_149, primals_150, primals_152, primals_153, primals_154, primals_155, primals_157, primals_158, primals_159, primals_160, primals_162, primals_163, primals_164, primals_165, primals_167, primals_168, primals_169, primals_170, primals_172, primals_173, primals_174, primals_175, primals_177, primals_178, primals_179, primals_180, primals_182, primals_183, primals_184, primals_185, primals_187, primals_188, primals_189, primals_190, primals_192, primals_193, primals_194, primals_195, primals_197, primals_198, primals_199, primals_200, primals_202, primals_203, primals_204, primals_205, primals_207, primals_208, primals_209, primals_210, primals_212, primals_213, primals_214, primals_215, primals_217, primals_218, primals_219, primals_220, primals_222, primals_223, primals_224, primals_225, primals_227, primals_228, primals_229, primals_230, primals_232, primals_233, primals_234, primals_235, primals_237, primals_238, primals_239, primals_240, primals_242, primals_243, primals_244, primals_245, primals_247, primals_248, primals_249, primals_250, primals_252, primals_253, primals_254, primals_255, primals_257, primals_258, primals_259, primals_260, primals_262, primals_263, primals_264, primals_265, primals_267, primals_268, primals_269, primals_270, primals_272, primals_273, primals_274, primals_275, primals_277, primals_278, primals_279, primals_280, primals_282, primals_283, primals_284, primals_285, primals_287, primals_288, primals_289, primals_290, primals_292, primals_293, primals_294, primals_295, primals_297, primals_298, primals_299, primals_300, primals_302, primals_303, primals_304, primals_305, primals_307, primals_308, primals_309, primals_310, primals_312, primals_313, primals_314, primals_315, primals_317, primals_318, primals_319, primals_320, primals_322, primals_323, primals_324, primals_325, primals_327, primals_328, primals_329, primals_330, primals_332, primals_333, primals_334, primals_335, primals_337, primals_338, primals_339, primals_340, primals_342, primals_343, primals_344, primals_345, primals_347, primals_348, primals_349, primals_350, primals_352, primals_353, primals_354, primals_355, primals_357, primals_358, primals_359, primals_360, primals_362, primals_363, primals_364, primals_365, primals_367, primals_368, primals_369, primals_370, primals_372, primals_373, primals_374, primals_375, primals_377, primals_378, primals_379, primals_380, primals_382, primals_383, primals_384, primals_385, primals_387, primals_388, primals_389, primals_390, primals_392, primals_393, primals_394, primals_395, primals_397, primals_398, primals_399, primals_400, primals_402, primals_403, primals_404, primals_405, primals_407, primals_408, primals_409, primals_410, primals_412, primals_413, primals_414, primals_415, primals_417, primals_418, primals_419, primals_420, primals_422, primals_423, primals_424, primals_425, primals_427, primals_428, primals_429, primals_430, primals_432, primals_433, primals_434, primals_435, primals_437, primals_438, primals_439, primals_440, primals_442, primals_443, primals_444, primals_445, primals_447, primals_448, primals_449, primals_450, primals_452, primals_453, primals_454, primals_455, primals_457, primals_458, primals_459, primals_460, primals_462, primals_463, primals_464, primals_465, primals_467, primals_468, primals_469, primals_470, primals_472, primals_473, primals_474, primals_475, primals_477, primals_478, primals_479, primals_480, primals_482, primals_483, primals_484, primals_485, primals_487, primals_488, primals_489, primals_490, primals_492, primals_493, primals_494, primals_495, primals_497, primals_498, primals_499, primals_500, primals_502, primals_503, primals_504, primals_505, primals_507, primals_508, primals_509, primals_510, primals_512, primals_513, primals_514, primals_515, primals_517, primals_518, primals_519, primals_520, primals_522, primals_523, primals_524, primals_525, primals_527, primals_528, primals_529, primals_530, primals_532, primals_533, primals_534, primals_535, primals_537, primals_538, primals_539, primals_540, primals_542, primals_543, primals_544, primals_545, primals_547, primals_548, primals_549, primals_550, primals_552, primals_553, primals_554, primals_555, primals_557, primals_558, primals_559, primals_560, primals_562, primals_563, primals_564, primals_565, primals_567, primals_568, primals_569, primals_570, primals_572, primals_573, primals_574, primals_575, primals_577, primals_578, primals_579, primals_580, primals_582, primals_583, primals_584, primals_585, primals_587, primals_588, primals_589, primals_590, primals_592, primals_593, primals_594, primals_595, primals_597, primals_598, primals_599, primals_600, primals_602, primals_603, primals_604, primals_605, primals_607, primals_608, primals_609, primals_610, primals_612, primals_613, primals_614, primals_615, primals_616, primals_618, primals_619, primals_620, primals_621, primals_623, primals_624, primals_625, primals_626, primals_628, primals_629, primals_630, primals_631, primals_633, primals_634, primals_635, primals_636, primals_638, primals_639, primals_640, primals_641, primals_643, primals_644, primals_645, primals_646, primals_648, primals_649, primals_650, primals_651, primals_653, primals_654, primals_655, primals_656, primals_657, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf94, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf111, buf112, buf113, buf114, buf115, buf116, buf117, buf118, buf119, buf120, buf121, buf122, buf123, buf124, buf125, buf126, buf127, buf128, buf129, buf130, buf131, buf132, buf133, buf134, buf135, buf136, buf137, buf138, buf139, buf140, buf141, buf142, buf143, buf144, buf145, buf146, buf147, buf148, buf149, buf150, buf151, buf152, buf153, buf154, buf155, buf156, buf157, buf158, buf159, buf160, buf161, buf162, buf163, buf164, buf165, buf166, buf167, buf168, buf169, buf170, buf171, buf172, buf173, buf174, buf175, buf176, buf177, buf178, buf179, buf180, buf181, buf182, buf183, buf184, buf185, buf186, buf187, buf188, buf189, buf190, buf191, buf192, buf193, buf194, buf195, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf203, buf204, buf205, buf206, buf207, buf208, buf209, buf210, buf211, buf212, buf213, buf214, buf215, buf216, buf217, buf218, buf219, buf220, buf221, buf222, buf223, buf224, buf225, buf226, buf227, buf228, buf229, buf230, buf231, buf232, buf233, buf234, buf235, buf236, buf237, buf238, buf239, buf240, buf241, buf242, buf243, buf244, buf245, buf246, buf248, buf249, buf250, buf251, buf252, buf254, buf255, buf256, buf257, buf258, buf259, buf260, buf261, buf262, buf263, buf264, buf265, reinterpret_tensor(buf266, (4, 2048), (2048, 1), 0), primals_658, )


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
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((728, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((728, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((728, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((728, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((1024, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((1024, 728, 1, 1), (728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1024, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((1536, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((1536, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((2048, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
